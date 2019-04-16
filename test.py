import argparse

from src.utils import Parameters
from src.experiment import Experiment
from src.logging import logger
from src.trainer import Trainer
from train import train


def get_parser():
    parser = argparse.ArgumentParser(description='Tests whether the code *runs* various combinations of the settings. Does not check for correctness.')
    parser.add_argument('--dataset_path', help="the directory where there is the train.txt and valid.txt files for the dataset", required=True)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    dataset_path = args.dataset_path
    embedding_size = 5
    neg_ratio = 3
    batch_size = 500
    max_iters = 2
    valid_interval = 5
    print_test = False
    checkpoint_file = "temp.pth"


    parameters = Parameters(
        embedding_size = embedding_size,
        neg_ratio = neg_ratio,
        batch_size = batch_size,
        max_iters = max_iters,
        valid_interval = valid_interval,
        print_test = print_test,
        type_loss = "softmargin",
        margin = 1.0
    )

    experiment = Experiment()
    experiment.construct_experiment(dataset_path, joint=True, typed_corrs=True)

    # ---- testing for vanilla models ---- #

    _ = train(experiment=experiment, parameters=parameters, method='transe_l2', joint=False, run_on_test=False)


    parameters.fact_loss = 'softmax'

    for method in ['transe_l2', 'transe_l1', 'complex', 'simple', 'distmult']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=False, run_on_test=False)
        print("Vanilla %s with %s runs! \n" % (method, parameters.fact_loss))

    parameters.fact_loss = 'full-softmax'

    for method in ['complex', 'simple', 'distmult']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=False, run_on_test=False)
        print("Vanilla %s with %s runs! \n" % (method, parameters.fact_loss))

    parameters.fact_loss = 'maxmargin'

    for method in ['transe_l1']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=False, run_on_test=False)
        print("Vanilla %s with %s runs! \n" % (method, parameters.fact_loss))


    # --- testing for joint models ---#

    parameters.fact_loss = 'softmax'
    parameters.type_loss = 'softmargin'

    for method in ['distmult', 'complex']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=True, run_on_test=False)
        print("Joint %s with %s runs! \n" % (method, parameters.fact_loss))

    parameters.fact_loss = 'full-softmax'
    parameters.type_loss = 'softmargin'

    for method in ['distmult', 'complex']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=True, run_on_test=False)
        print("Joint %s with %s runs! \n" % (method, parameters.fact_loss))


    # --- testing for typed negatives ---#

    parameters.fact_loss = "softmax"
    parameters.type_loss = "softmargin"
    parameters.type_ratios = [0.8, 0.0, 0.2]

    for method in ['distmult']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=True, run_on_test=False)
        print("Joint %s with %s and typed negatives runs! \n" % (method, parameters.fact_loss))

    for method in ['distmult']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=False, run_on_test=False)
        print("Vanilla %s with %s and typed negatives runs! \n" % (method, parameters.fact_loss))

    parameters.fact_loss = "full-softmax"
    parameters.type_loss = "softmargin"
    parameters.type_ratios = [0.8, 0.0, 0.2]

    for method in ['distmult']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=True, run_on_test=False)
        print("Joint %s with %s and typed negatives runs! \n" % (method, parameters.fact_loss))

    for method in ['distmult']:
        _ = train(experiment=experiment, parameters=parameters, method=method, joint=False, run_on_test=False)
        print("Vanilla %s with %s and typed negatives runs! \n" % (method, parameters.fact_loss))


if __name__ == '__main__':
    main()
