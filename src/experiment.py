import numpy as np

from collections import defaultdict
from .logging import *
from .utils import make_head_and_tail_dicts, make_head_and_tail_dicts_from_labels, turn_head_and_tail_dicts_into_arr, convert_triples_list_to_arr

class Experiment(object):
    '''
    Class for encapsulating the dataset that the models will run.

    Attributes
    ----------
    train_triples: list(int)
    valid_triples: list(int)
    test_triples: list(int)
    n_entities: int
    n_relations: int
    entities_dict: dict(string : int)
        dictionary mapping the entity names to indices
    relations_dict: dict(string : int)
        dictionary mapping relation names to indices
    known_obj_triples: dict((int, int): list(int))
        dictionary mapping pairs of head and rel ids to tail ids they occur with in
        training, validation or test sets.
    known_subj_triples: dict((int, int): list(int))
        dictionary mapping pairs of rel and tail ids to head ids they occur with in
        training, validation or test sets.
    head_dict (optional): dict(int: set(int))
        dictionary mapping relation ids to the set of entity ids which occurred in the
        training data as the head of the relation
    tail_dict (optional): dict(int: set(int))
        dictionary mapping relation ids to the set of entity ids which occurred in the
        training data as the tail of the relation
    head_dict_arr (optional): dict(int: np.array)
        same content as head_dict, but the values are binary numpy arrays of size n_entities,
        where the index corresponding to an entity is 1 if that entity occurred as the
        head of the relation in the training triples, and 0 otherwise
    tail_dict_arr (optional): dict(int: np.array)
        same content as tail_dict, but the values are binary numpy arrays of size n_entities,
        where the index corresponding to an entity is 1 if that entity occurred as the
        tail of the relation in the training triples, and 0 otherwise
    '''
    def __init__(self):
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.n_entities = 0
        self.n_relations = 0
        self.entities_dict = {}
        self.relations_dict = {}
        self.known_obj_triples = defaultdict(set)
        self.known_subj_triples = defaultdict(set)
        self.head_dict = None
        self.tail_dict = None
        self.head_dict_arr = None
        self.tail_dict_arr = None

    def load_triples(self, filename, next_ent_idx, next_rel_idx):
        '''Helper function to populate the experiment object.

        Parameters
        ----------
        filename: string
        next_ent_idx: int
            the next free index to assign to the first previously unseen entity
        next_rel_idx: int
            the next free index to assign to the first previoously unseen relation

        Returns
        -------
        list(tuple)
            list of triples where each tuple consists of entity id, relation id, entity id
        int
            next free index for entities
        int
            next free index for relations
        '''
        triples = []
        with open(filename, 'r') as f:
            for line in f:
                head, rel, tail = line.strip().split('\t')

                if head not in self.entities_dict:
                    self.entities_dict[head] = next_ent_idx
                    next_ent_idx += 1

                if tail not in self.entities_dict:
                    self.entities_dict[tail] = next_ent_idx
                    next_ent_idx += 1

                if rel not in self.relations_dict:
                    self.relations_dict[rel] = next_rel_idx
                    next_rel_idx += 1

                head_idx = self.entities_dict[head]
                rel_idx = self.relations_dict[rel]
                tail_idx = self.entities_dict[tail]

                self.known_obj_triples[(head_idx, rel_idx)].add(tail_idx)
                self.known_subj_triples[(rel_idx, tail_idx)].add(head_idx)

                triples.append((head_idx, rel_idx, tail_idx))

        return triples, next_ent_idx, next_rel_idx

    def construct_experiment(self, pathname, joint=False, typed_corrs=False, type_files_path=None):
        '''
        Construct an experiment object from a given path.

        Parameters
        ----------
        pathname: string
            must contain train.txt, valid.txt and test.txt
        joint: bool
            True if the extra structures for joint training should be created, False otherwise
        typed_corrs: bool
            True if the extra structures for typed negative generation should be created, False otherwise
        type_files_path: string
            If this is given, the data structures for joint and typed negatives are created
            using the explicit labels given in this path, and not heuristically
        '''

        train_filename = pathname + '/train.txt'
        valid_filename = pathname + '/valid.txt'
        test_filename = pathname + '/test.txt'

        self.train_triples, next_ent_idx, next_rel_idx = self.load_triples(train_filename, 0, 0)
        self.valid_triples, next_ent_idx, next_rel_idx = self.load_triples(valid_filename, next_ent_idx, next_rel_idx)
        self.test_triples, _, _ = self.load_triples(test_filename, next_ent_idx, next_rel_idx)

        self.n_entities = len(self.entities_dict)
        self.n_relations = len(self.relations_dict)

        # convert these from dictionaries of sets to dictionaries of lists
        self.known_obj_triples = {key: list(value) for key, value in self.known_obj_triples.items()}
        self.known_subj_triples = {key: list(value) for key, value in self.known_subj_triples.items()}

        if (joint or typed_corrs) and not type_files_path:
            self.head_dict, self.tail_dict = make_head_and_tail_dicts(self.train_triples)

        elif (joint or typed_corrs) and type_files_path:
            ent2idx = reverse_dict(self.entities_dict)
            rel2idx = reverse_dict(self.relations_dict)
            self.head_dict, self.tail_dict = make_head_and_tail_dicts_from_labels(type_files_path, ent2idx, rel2idx)

        if typed_corrs:
            self.head_dict_arr, self.tail_dict_arr = turn_head_and_tail_dicts_into_arr(self.head_dict, self.tail_dict)

        # turn train triples into numpy array
        self.train_triples = convert_triples_list_to_arr(self.train_triples)

        logger.info("Number of entities: {}".format(self.n_entities))
        logger.info("Number of relations: {}".format(self.n_relations))
        logger.info("Size of training set: {}".format(self.train_triples.shape[0]))
        logger.info("Size of validation set: {}".format(len(self.valid_triples)))
        logger.info("Size of test set: {}".format(len(self.test_triples)))

        assert(self.head_dict_arr is None or type(self.head_dict_arr[0]) == np.ndarray), 'Expected values in the head_dict_arr to be np.ndarray, found %s instead' % str(type(self.head_dict_arr[0]))
