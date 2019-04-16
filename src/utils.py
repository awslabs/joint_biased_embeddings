import torch as t
from torch import nn
from random import choice, randint
from collections import defaultdict
import numpy as np
from .logging import logger

FACT_LOSSES = ['softmax', 'full-softmax', 'softmargin', 'maxmargin', 'hinge']
TYPE_LOSSES = ['softmargin', 'maxmargin', 'hinge']


def get_device():
    return t.device("cuda" if t.cuda.is_available() else "cpu")

#  _          _                        __              _
# | |__   ___| |_ __   ___ _ __ ___   / _| ___  _ __  | | ___  ___ ___  ___  ___
# | '_ \ / _ \ | '_ \ / _ \ '__/ __| | |_ / _ \| '__| | |/ _ \/ __/ __|/ _ \/ __|
# | | | |  __/ | |_) |  __/ |  \__ \ |  _| (_) | |    | | (_) \__ \__ \  __/\__ \
# |_| |_|\___|_| .__/ \___|_|  |___/ |_|  \___/|_|    |_|\___/|___/___/\___||___/
#              |_|

def get_correct_and_incorrect_scores(scores, neg_ratio):
    '''
    Splits the scores assigned to correct and incorrect scores.

    Parameters
    ----------
    scores: torch.FloatTensor
        scores assigned by the model to each triple in the batch
    neg_ratio: int
        ratio of generated negatives per positive example

    Returns
    -------
    tuple with correct and incorrect scores shaped (batchsize,) and (batchsize, neg_ratio)

    Example
    -------
    batch size: 2, neg ratio 3:

    batch input example:
    (paris, capital-of, france)
    (everest, located-in, asia)

    with corrupted 3 corrupted samples the batch becomes:
    (paris, capital-of, france)
    (everest, located-in, asia)
    (paris, capital-of, the room (movie)
    (barack obama, located-in, asia)
    (united nations, capital-of, france)
    (everest, located-in, kebab (dish)
    (paris, capital-of, world war 2)
    (pnemonia, located-in, asia)

    e.g. scores passed into this function:
    torch.tensor([3.4, 5.6, 2.2, 1.4, 0.5, -3.4, 1.2, 3.3])

    get_correct_and_incorrect_scores takes this and turns it into:
    (torch.tensor([3.4, 5.6]), torch.tensor([[2.2, 0.5, 1.2],
    					                     [1.4, -3.4, 3.3]]))

    So that output[0][i] containts the score for the original triple i, and
    output[1][i] contains scores for all its corrupted versions.
    '''
    num_correct = scores.size()[0]/(neg_ratio+1)
    # check that the number of correct examples is indeed an integer
    assert abs(num_correct - int(num_correct)) < 1e-3, "The size of scores and the negative ratio is not compatible."
    num_correct = int(num_correct)
    scores_correct = scores[:num_correct]
    scores_incorrect = scores[num_correct:].view(num_correct, neg_ratio)

    return (scores_correct, scores_incorrect)

def get_head_and_tail_scores(scores, neg_ratio):
    '''
    Splits the scores assigned to head and tail corrupted triples.

    Notes
    -----
    * Not currently used

    Parameters
    ----------
     scores: torch.FloatTensor
         scores assigned by the model to each triple in the batch
     neg_ratio: int
         ratio of generated negatives per positive example

     Returns
     -------
     tuple with head corrupted and tail corrupted scores. The scores for the correct
     triples are repeated in the two sets.
     '''
    batchsize = int(scores.shape[0]/(neg_ratio + 1))
    half_neg = int(neg_ratio/2)*batchsize
    scores_head = scores[:(half_neg + batchsize)]
    scores_tail = t.cat((scores[:batchsize], scores[half_neg + batchsize:]))
    return scores_head, scores_tail

#  _          _                        __                                        _                      _
# | |__   ___| |_ __   ___ _ __ ___   / _| ___  _ __    _____  ___ __   ___ _ __(_)_ __ ___   ___ _ __ | |_
# | '_ \ / _ \ | '_ \ / _ \ '__/ __| | |_ / _ \| '__|  / _ \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __|
# | | | |  __/ | |_) |  __/ |  \__ \ |  _| (_) | |    |  __/>  <| |_) |  __/ |  | | | | | | |  __/ | | | |_
# |_| |_|\___|_| .__/ \___|_|  |___/ |_|  \___/|_|     \___/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|
#              |_|                                              |_|

def make_head_and_tail_dicts(data):
     '''
     Make two dictionaries where the keys are relations and values are sets consisting
     of all the entities that occurred as the head and the tail of that relation respectively.

     Parameters
     ----------
     data: list
         dataset to be processed. This should be a numpy array of size (n_datapoints, 3)

     Returns
     -------
     dict(set), dict(set)
         A tuple where both indeces contain dictionaries which has relations as keys.
         The first dictionary has head entities of the relation as values, and the second
         dictionary has tail entities.
         '''
     head_dict = defaultdict(set)
     tail_dict = defaultdict(set)

     for e1, r, e2 in data:

         head_dict[r].add(e1)
         tail_dict[r].add(e2)

     return (dict(head_dict), dict(tail_dict))

def turn_head_and_tail_dicts_into_arr(head_dict, tail_dict):
    '''
    Converts head and tail dictionaries into dictionaries where the values
    are numpy arrays rather than sets.

    Notes
    -----
    This step speeds up the computation significantly for generating correctly
    typed negatives, but is not needed otherwise.

    Parameters
    ----------
    head_dict: dict(set)
        Dictionary with relation indeces as keys and sets of entity indeces as values
    tail_dict: dict(set)
        Dictionary with relation indeces as keys and sets of entity indeces as values

    Returns
    -------
    dict(np.array), dict(np.array)
    '''
    head_dict_arr = {}
    tail_dict_arr = {}

    for r in head_dict: # r should also be in tail_dict
        h_indeces = np.empty(len(head_dict[r]), dtype=np.uint32)
        t_indeces = np.empty(len(tail_dict[r]), dtype=np.uint32)

        for i, idx in enumerate(head_dict[r]):
            h_indeces[i] = np.uint32(idx)

        for i, idx in enumerate(tail_dict[r]):
            t_indeces[i] = np.uint32(idx)

        head_dict_arr[r] = h_indeces
        tail_dict_arr[r] = t_indeces

    return (head_dict_arr, tail_dict_arr)

def make_ents2type_dict(filename, ent2idx):
    '''
    Constructs a dictionary mapping entities to their explicit types.

    Parameters
    ----------
    filename: string
        format: entity	type_1	...	type_n
    ent2idx: dict(int)
        dictionary mapping entity names to the ids

    Returns
    -------
    dict(set)
        keys are entity ids and the values are sets of type labels
    '''
    ent_types = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().split('\t')
            ent_idx = ent2idx[line[0]]
            ent_types[ent_idx] = set(line[1:])

    return ent_types

def make_type2ents_dict(filename, ent2idx):
    '''
    Constructs a dictionary mapping types to their corresponding entities.

    Parameters
    ----------
    filename: string
        format: entity	type_1	...	type_n
    ent2idx: dict(int)
        dictionary mapping entity names to the ids

    Returns
    -------
    dict(set)
        keys are type labels and values are sets of entity ids.
    '''
    ent_types = defaultdict(set)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] in ent2idx:
                ent_idx = ent2idx[line[0]]
                for type in line[1:]:
                    ent_types[type].add(ent_idx)

    return ent_types

def make_type_dicts_rels(filename, rel2idx):
    '''
    Constructs two dictionaries, one that maps each relation to the type of its head
    arguments and the other to the type of its tail arguments

    Parameters
    ----------
    filename: string
        expected format: relation	head type	tail type
    rel2idx: dict(int)
        dictionary mapping relation names to ids

    Returns
    -------
    dict(string), dict(string)
    '''
    head_types = {}
    tail_types = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] in rel2idx:
                rel_idx = rel2idx[line[0]]
                head_types[rel_idx] = line[1]
                tail_types[rel_idx] = line[2]

    return (head_types, tail_types)

def make_head_and_tail_dicts_from_labels(foldername, ent2idx, rel2idx):
    '''
    Constructs head_dict and tail_dict from explicit type labels

    Parameters
    ----------
    foldername: string
        Should contain the two files: '/entity2type.txt', '/relation_specific.txt'
    ent2idx: dict(int)
        dictionary mapping entity names to indices
    rel2idx: dict(int)
        dictionary mapping relation names to indices

    Returns
    -------
    dict(set), dict(set)
        A tuple where both indeces contain dictionaries which has relations as keys.
        The first dictionary entities matcihng the head type of the relation as values,
        and the second dictionary has entities matching the tail type.
    '''
    entfile = foldername + '/entity2type.txt'
    relfile = foldername + '/relation_specific.txt'

    type2ent = make_type2ents_dict(entfile, ent2idx)
    head_types, tail_types = make_type_dicts_rels(relfile, rel2idx)

    head_dict = {}
    tail_dict = {}

    for rel in head_types:
        head_type = head_types[rel]
        tail_type = tail_types[rel]
        head_dict[rel] = type2ent[head_type]
        tail_dict[rel] = type2ent[tail_type]

    return (head_dict, tail_dict)

def reverse_dict(orig_dict):
    '''
    Reverses a dictionary where each value is unique. If any value is non unique,
    it overrides the previous entry.
    '''
    rev_dict = {}
    for key, value in orig_dict.items():
        rev_dict[value] = key

    return rev_dict

def convert_triples_list_to_arr(triples):
    '''
    Convert a list of triples to a numpy array of shape (dataset_size, 3)
    '''
    triples_arr = np.zeros((len(triples), 3), dtype=np.long)
    for i, (e1, r, e2) in enumerate(triples):
        triples_arr[i, 0] = e1
        triples_arr[i, 1] = r
        triples_arr[i, 2] = e2

    return triples_arr


#  _          _                        __              _           _       _     _
# | |__   ___| |_ __   ___ _ __ ___   / _| ___  _ __  | |__   __ _| |_ ___| |__ (_)_ __   __ _
# | '_ \ / _ \ | '_ \ / _ \ '__/ __| | |_ / _ \| '__| | '_ \ / _` | __/ __| '_ \| | '_ \ / _` |
# | | | |  __/ | |_) |  __/ |  \__ \ |  _| (_) | |    | |_) | (_| | || (__| | | | | | | | (_| |
# |_| |_|\___|_| .__/ \___|_|  |___/ |_|  \___/|_|    |_.__/ \__,_|\__\___|_| |_|_|_| |_|\__, |
#              |_|                                                                       |___/

def check_percent_typed_examples(data, head_dict, tail_dict, n_entities):
    '''
    Roughtly approximates what percent of randomly generated examples would
    end up typechecking.

    Notes
    -----
    This function is not needed for training/evaluation, but might be useful as
    a step to determine the hyperparameters before the training.
    '''
    total_tries = 0
    hits = 0
    for i in range(data.shape[0]):
        r = data[i,1]
        for _ in range(100):
            total_tries += 1
            randent = randint(0, n_entities)
            if choice([True, False]):
                if randent in head_dict[r]:
                    hits += 1
            else:
                if randent in tail_dict[r]:
                    hits += 1

    return hits/total_tries



def generate_type_labels_for_full_softmax(r_idx, ent_dict_arr, n_entities):
    '''
    Calculates the type labels for the batch for full softmax loss

    Parameters
    ----------
    r_idx: numpy.array
        relation indexes with shape: (batch_size,)
    ent_dict_arr: dict(numpy.array)
        head_dict_arr or tail_dict_arr generated by turn_head_and_tail_dicts_into_arr

    Returns
    -------
    numpy.array
        This is an array of values: 1, -1, shape: (batch_size, n_entities) where
        the ith row is ent_dict_arr[r_idx[i]]
    '''
    batch_size = r_idx.shape[0]
    type_labels = np.zeros((batch_size, n_entities), dtype=np.float32)
    for i in range(batch_size):
        type_labels[i,r_idx[i]] = 1

    return type_labels

#    ___                               _
#   / _ \__ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___
#  / /_)/ _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|
# / ___/ (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \
# \/    \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/

class Parameters(object):
    '''Container for hyperparameters for training the model.

    Attributes
    ----------
    embedding_dim: int
        embedding dimension
    batch_size: int
        size of batches before corrupted negatives are added
    max_iters: int
        maximum epochs to train the model
    learning_rate: float
    valid_interval: int
        During training, the interval in epochs between every validation run
    fact_loss: string
        The loss to train the fact module with.
        options:  'maxmargin', 'hinge', 'softmargin', 'softmax', 'full-softmax'
    type_loss: string
        The loss to train the type module with.
        options: 'maxmargin', 'hinge', 'softmargin'
    policy: string
        Optimization policy.
        options: 'adagrad', 'adam'
    early_stop_with: string
        Which metric to use to do early stopping
        options: 'hits@1', 'hits@3', 'hits@10', 'mrr', 'rmrr'
    print_test: bool
        True if the final results should be printed on the test set, False otherwise
    checkpoint_file (optional): string
        Filename to save the model after each validation step
    type_ratios: list(float)
        Relative weights of the three type generation strategies. See batching.py for
        details
    loss_ratio: float
        Weight to multiply the type loss with before backprop
    fact_module: string
        Name of the fact module.
        options: 'DistMult', 'ComplEx', 'SimplE', 'TransE_L1', 'TransE_L2'
    type_module (optional): string
        Name of the type module
        options: 'DistMult', 'ComplEx', 'SimplE', 'TransE_L1', 'TransE_L2'
    lmbda: float
        L2 regularization value
    margin: float
        margin to use for maxmargin and hinge losses
    batch_params: dict
        A dictionary with two keys 'batch_size' and 'num_workers'. Increasing both
        batch_size and num_workers might speed up the training due to multiprocessing
        when preparing the batches. Increasing the batch size here will multiply the
        effective batch size during training with that amount.
        '''
    def __init__(self, embedding_size = 200, batch_size = 500, neg_ratio = 20,
        max_iters = 5000, learning_rate = 0.01, valid_interval = 5, fact_loss='softmax',
        type_loss='softmargin', policy = 'adam', early_stop_with = 'hits@10',
        checkpoint_file=None, print_test = False, type_ratios=[1., 0., 0.],
        loss_ratio=0.5, lmbda = 0.0, margin=0, batch_params={'batch_size':1, 'num_workers':0}):

        self.fact_loss = fact_loss
        self.type_loss = type_loss
        self.lmbda = lmbda
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.neg_ratio = neg_ratio
        self.valid_interval = valid_interval
        self.policy = policy
        self.early_stop_with = early_stop_with
        self.checkpoint_file = checkpoint_file
        self.print_test = print_test
        self.margin = margin
        self.batch_params = batch_params
        self.type_ratios = type_ratios
        self.loss_ratio = loss_ratio

#  _                   _
# | | ___   __ _  __ _(_)_ __   __ _
# | |/ _ \ / _` |/ _` | | '_ \ / _` |
# | | (_) | (_| | (_| | | | | | (_| |
# |_|\___/ \__, |\__, |_|_| |_|\__, |
#          |___/ |___/         |___/

def log_results(results):
    '''
    Pretty prints the given results.

    Parameters
    ---------
    results: dict
        expected keys are 'hits@1', 'hits@3', 'hits@10', 'mrr', 'rmrr', and
        the expected values are float
    '''
    logger.info("hits@1\t\thits@3\t\thits@10\t\tMRR\t\tRMRR")
    logger.info("{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(
            results['hits@1'], results['hits@3'], results['hits@10'], results['mrr'], results['rmrr']))

def log_parameters(parameters):
    # TODO
    '''
    Pretty prints the given parameters.

    Parameters
    ---------
    parameters: Parameters
    '''
    raise NotImplementedError()
