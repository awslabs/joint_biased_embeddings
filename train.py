# Takes the name of the model, the dataset, a list of loss ratios, a list of
# negative ratios, a list of embedding sizes and a list of type ratios, and does
# a gridsearch over all of them, and prints the results of the gridsearch.
# It doesn't check the test set during the gridsearch, but if the option is given,
# reruns the model with the best parameters and prints out the results on the test
# set.

# NOTE: make sure to add flags --joint and --typed_negatives if you want the training to
# take into account the parameters given to --type_ratios and --loss_ratios etc.
# if you run it on a previously generated experiment with different flags, this will
# cause an error to be thrown.

import argparse
import pickle
import src
from src.logging import logger
from src.utils import Parameters, FACT_LOSSES, TYPE_LOSSES
from src.experiment import Experiment
from src.modules import create_module, MODULES, BILINEAR_MODULES
from src.trainer import Trainer


def get_experiment_summary(embedding_size, neg_ratio, batch_size, learning_rate, loss_ratio, type_ratios):
    summary = ["embedding size: {}".format(embedding_size)]
    summary.append("batch size: {}".format(batch_size))
    summary.append("negatives ratio: {}".format(neg_ratio))
    summary.append("loss ratio: {}".format(loss_ratio))
    summary.append("typed negatives ratio: {}".format(type_ratios))
    summary.append("learning rate {}".format(learning_rate))

    return '\n'.join(summary)


def print_results(dataset_name, results):
    print("Results on %s" % dataset_name)
    print("hits@1\t\thits@3\t\thits@10\t\tMRR\t\tRMRR")
    print("{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(
            results['hits@1'], results['hits@3'], results['hits@10'], results['mrr'], results['rmrr']))


def train(experiment, parameters, method, joint, run_on_test=False):
    experiment_summary = get_experiment_summary(parameters.embedding_size, parameters.neg_ratio, parameters.batch_size,
                                                parameters.learning_rate, parameters.loss_ratio, parameters.type_ratios)


    module = create_module(experiment, method, joint, parameters)

    logger.info(" \n------------------ \n")
    logger.info("Running model with:\n")
    logger.info(experiment_summary)

    trainer = Trainer(module)
    valid_result = trainer.fit(experiment, parameters)

    print(experiment_summary)
    print_results('validation set', valid_result)

    if run_on_test:
        test_results = trainer.test(experiment, parameters.checkpoint_file)
        print_results('test set', test_results)
    else:
        test_results = None

    return valid_result, test_results


def get_parameters(args):
    parameters = Parameters(
                    embedding_size = args.embedding_size,
                    batch_size = args.batch_size,
                    neg_ratio = args.negative_ratio,
                    learning_rate = args.learning_rate,
                    loss_ratio = args.loss_ratio,
                    type_ratios = [1.0 - args.type_ratio, 0.0, args.type_ratio],
                    valid_interval = args.valid_interval,
                    fact_loss = args.fact_loss,
                    type_loss = args.type_loss,
                    policy = args.policy,
                    early_stop_with = args.validate_with,
                    checkpoint_file = args.checkpoint_file,
                    print_test = False,
                    max_iters = args.max_epochs
    )

    return parameters


def get_parser():
    parser = argparse.ArgumentParser(description='Train on given module and dataset, and with given parameter values')
    parser.add_argument('method', choices=MODULES.keys())
    parser.add_argument('--dataset_path', help="the directory where there is the train.txt and valid.txt files for the dataset")
    parser.add_argument('--load_experiment_from', help="the pickle file to load an already constructed experiment")
    parser.add_argument('--embedding_size', default=200, type=int, help="Embedding dimension")
    parser.add_argument('--batch_size', default=500, type=int, help="Batch size")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--negative_ratio', default=50, type=int, help="Ratio of generated negative training examples per positive example")
    parser.add_argument('--loss_ratio', default=0.25, type=float, help="Weight for the type module loss")
    parser.add_argument('--type_ratio', default=0.0, type=float, help="Ratio of typed-corrupted examples to randomly corrupted examples")
    # TODO: construct the experiments with explicit type labels if they're given
    # parser.add_argument('--type_file', help="path to type labels for the entities and relations in the dataset")
    parser.add_argument('--policy', default='adam', choices=['adam', 'adagrad'], help='policy to train the models with')
    parser.add_argument('--run_on_test', default=False, action='store_true', help="Run every model against a test set")
    parser.add_argument('--validate_with', default='hits@10', choices=['hits@1', 'hits@3', 'hits@10', 'mrr', 'rmrr'], help="The metric to do the early stopping and to pick the best parameters with")
    parser.add_argument('--joint', default=False, action='store_true', help="Create the experiments for joint training, and run the models as joint models")
    parser.add_argument('--typed_negatives', default=False, action='store_true', help="Create the experiments for typed negatives generation")
    parser.add_argument('--save_experiment_to', help='name of the file to pickle and save the experiment after its created')
    parser.add_argument('--valid_interval', default=5, type=int, help="number of epoch to wait between each validation step")
    parser.add_argument('--fact_loss', default='softmax', choices=FACT_LOSSES)
    parser.add_argument('--type_loss', default='softmargin', choices=TYPE_LOSSES)
    parser.add_argument('--checkpoint_file', default='temp.pth', help="Filename to save the parameters of the model during training")
    parser.add_argument('--max_epochs', default=5000, type=int, help="Maximum number of epochs to run the training")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.joint and args.method not in BILINEAR_MODULES:
        raise ValueError('Joint Models are only supported for bilinear models: %s' % BILINEAR_MODULES.keys())

    # Construct or load the experiment
    if args.load_experiment_from:
        print("Loading experiment from {}".format(args.load_experiment_from))
        experiment = pickle.load(open(args.load_experiment_from, "rb"))

    elif args.dataset_path:
        experiment = Experiment()
        print("Constructing experiment...")
        experiment.construct_experiment(args.dataset_path, args.joint, args.typed_negatives)
        print("Experiment constructed!")

        if args.save_experiment_to:
            print("Saving experiment to {}".format(args.save_experiment_to))
            pickle.dump(experiment, open(args.save_experiment_to, "wb"))

    else:
        raise ValueError("Either dataset_name or load_experiment_from must be given")

    parameters = get_parameters(args)

    _ = train(experiment, parameters, args.method, args.joint, args.run_on_test)


if __name__ == '__main__':
    main()
