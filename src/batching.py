
import torch as t
from random import sample
import numpy as np
import warnings
from math import ceil
from .utils import generate_type_labels_for_full_softmax

class TriplesDataset(t.utils.data.Dataset):
    '''
    A pytorch Dataset object that generates batches to consume for the training.
    It batches the triples together, generates corrupted negatives and labels the
    triples accordingly.

    Each item this generator returns is a batch of given size rather than an individual
    triple. This is for performance reasons: generating corrupted triples for the
    entire minibatch at once is faster than generating them iteratively.

    Attributes
    ----------
    data: list of tuples of the form (int, int, int)
    batch_size: int
    neg_ratio: int
    n_entities: int
    '''
    def __init__(self, data, n_entities, batch_size, neg_ratio=1):
        np.random.shuffle(data)
        self.data = data
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.n_entities = n_entities

    def __len__(self):
        return ceil(self.data.shape[0] / self.batch_size)

    def __getitem__(self, index):
        '''
        Gets the next batch. This is the main function to implement for
        torch.utils.data.Dataset class instance.

        Parameters
        ----------
        index: int

        Returns
        -------
        (torch.Tensor, torch.Tensor)
        '''

        triples_pos = self.data[index * self.batch_size : min((index + 1) * self.batch_size, self.data.shape[0])]
        # this is the batch size after the negatives has been added
        batchsize = triples_pos.shape[0]
        # initialize the labels to all 1s
        values = np.ones((batchsize*(self.neg_ratio + 1),), dtype=np.float32)
        if self.neg_ratio > 0:
            # change the labels for corrupted triples to 0s
            values[batchsize:] = np.zeros_like(values[batchsize:])
            triples = np.tile(triples_pos, (self.neg_ratio + 1, 1))
            # generate all the candidate corrupted entities at once
            rand_ents = np.random.randint(low=0, high=self.n_entities, size=(batchsize*self.neg_ratio,))
            # generate random choices for corrupting the head or the tail entity
            rand_choices =  np.random.random(batchsize * self.neg_ratio) < 0.5

            for i in range(batchsize*self.neg_ratio):
                if rand_choices[i]: #TAILS! corrupt the tail entity
                    triples[batchsize+i,2] = rand_ents[i]
                else: # HEADS! corrupt the head entity
                    triples[batchsize+i,0] = rand_ents[i]

        else:
            triples = triples_pos

        assert triples.shape[0] == values.shape[0], "The shape of the triples and values mismatch"
        return (t.from_numpy(triples), t.from_numpy(values))

class TriplesDatasetFullSoftmax(TriplesDataset):
    '''
    A pytorch Dataset object for training with full-softmax + NLL

    Attributes
    ----------
    data: list of tuples of the form (int, int, int)
    batch_size: int
    n_entities: int
    joint: bool
        True if training is done both on fact and type labels, False otherwise
    head_dict_arr: dict(numpy.array)
        a dictionary with relation indexes as keys and a binary array as values, where
        head_dict_arr[r][i] = 1 if entity i ever occurred as the head entity of r
    tail_dict_arr: dict(numpy.array)

    Returns
    -------
    torch.Tensor (if not joint), or (torch.Tensor, (torch.Tensor, torch.Tensor)) (if joint)

    '''
    def __init__(self, indexes, n_entities, batch_size, joint=False, head_dict_arr=None, tail_dict_arr=None):
        np.random.shuffle(indexes)
        self.data = indexes
        self.batch_size = batch_size
        self.n_entities = n_entities
        self.joint = joint
        self.head_dict_arr = head_dict_arr
        self.tail_dict_arr = tail_dict_arr

    def __getitem__(self, index):
        # take a slice of the training data
        triples = self.data[index * self.batch_size : min((index + 1) * self.batch_size, self.data.shape[0])]

        if self.joint:
            r = triples[:,1]
            type_labels_head_corrupted = generate_type_labels_for_full_softmax(r, self.head_dict_arr, self.n_entities)
            type_labels_tail_corrupted = generate_type_labels_for_full_softmax(r, self.tail_dict_arr, self.n_entities)
            return (t.from_numpy(triples), (t.from_numpy(type_labels_head_corrupted), t.from_numpy(type_labels_tail_corrupted)))

        else:
            return t.from_numpy(triples)

class TriplesDatasetSeparate(TriplesDataset):
    #TODO: implement
    '''TO BE IMPLEMENTED: torch Dataset class that generates head corrupted negatives
    and tail corrupted negatives seperately.'''
    def __init__(self, indexes, n_entities, joint=False, head_dict_arr=None, tail_dict_arr=None, batch_size=100):
        raise NotImplementedError()
        np.random.shuffle(indexes)
        self.data = indexes
        self.batch_size = batch_size
        self.n_entities = n_entities
        self.head_dict_arr = head_dict_arr
        self.tail_dict_arr = tail_dict_arr

    def __getitem__(self, index):
        raise NotImpementedError()

class TriplesDatasetTyped(TriplesDataset):
    '''
    Dataset class that will generate the negative examples with three different strategies.

    Strategies are:

    1. corrupt the head or the tail with a random entity

    2. corrupt both the head and the tail with random entities

    3. corrupt the head or the tail with an entity of correct type

    When generating typed examples, it is much faster to use a numpy array instead of set.
    For this, the head_dict_arr and tail_dict_arr should be passed. If these aren't given,
    head_dict and tail_dict will be used instead.

    Attributes
    ----------
    type_ratios: list of floats
        The relative weights for the three negative sampling strategies. Should have length 3. The
        values will be normalized to sum to 1.
    head_dict: dict(set)
        Dictionary with relation ids as keys, and sets of entity ids as values. e is in head_dict[r]
        if e ever occurred as the head of r.
    tail_dict: dict(set)
    head_dict_arr: dict(np.array)
        Contains the same information as head_dict, but instead of set each value is a
        boolean numpy array of the same size as n_entities
    tail_dict_arr: dict(np.array)
    '''
    def __init__(self, indexes, n_entities, batch_size, neg_ratio, head_dict=None, tail_dict=None, joint=False, head_dict_arr=None, tail_dict_arr=None, type_ratios=[1., 0., 0.]):

        super(TriplesDatasetTyped, self).__init__(indexes, n_entities, batch_size, neg_ratio)
        assert(head_dict_arr is None or type(head_dict_arr[0]) == np.ndarray), 'Expected values in the head_dict_arr to be np.ndarray, found %s instead' % str(type(head_dict_arr[0]))

        if len(type_ratios) != 3:
            raise ValueError("Length of type_ratios need to be 3")
        # type ratios = [one rnd corrupted, both rnd corrupted, one typed corrupted] -- need to add upto one.
        self.type_ratios = np.asarray(type_ratios)
        if not np.sum(self.type_ratios) == 0.0:
            self.type_ratios = self.type_ratios/np.sum(self.type_ratios)
        self.head_dict = head_dict
        self.tail_dict = tail_dict
        self.head_dict_arr = head_dict_arr
        self.tail_dict_arr = tail_dict_arr
    
        self.joint = joint

        if not head_dict_arr:
            warnings.warn("Empty head_dict_arr!")

        if not head_dict_arr and type_ratios[2] > 0:
            warnings.warn("Typed negatives will be generated using dictionaries of sets -- this is inefficient. It's recommended that you pass them as array dictionaries as head_dict_arr and tail_dict_arr")

    def __getitem__(self, index):
        assert(self.head_dict_arr is None or type(self.head_dict_arr[0]) == np.ndarray), 'Expected values in the head_dict_arr to be np.ndarray, found %s instead' % str(type(self.head_dict_arr[0]))
        # take a slice of the training data
        triples_pos = self.data[index * self.batch_size : min((index + 1) * self.batch_size, self.data.shape[0])]
        batchsize = triples_pos.shape[0]

        # initialized all labels as false, correct to true as you go
        values = np.zeros((batchsize*(self.neg_ratio + 1), 2), dtype=np.float32)
        values[:batchsize] = 1.0 + values[:batchsize]

        # tile the positive triples as many times as needed for the full batch
        triples = np.tile(triples_pos,(self.neg_ratio + 1, 1))

        assert values.shape[0] == triples.shape[0]

        if np.sum(self.type_ratios) == 0 and self.neg_ratio != 0:
            raise ValueError("if type_ratios add up to 0, then neg_ratio must also be 0")

        # if you do not need to generate negatives you can stop and return the batch now.
        if self.neg_ratio == 0:
            return (t.from_numpy(triples), t.from_numpy(values))

        # generate all the randomness you might need for rnd_1 and rnd_2 in advance
        # this determines which of the three strategies to take for corrupting the triple
        choices = np.random.choice(3, batchsize*self.neg_ratio, p=self.type_ratios)

        # coin_flips determines whether to corrupt the head or the tail
        coin_flips = np.random.random(batchsize * self.neg_ratio) < 0.5

        # generate candidate corrupted entities in advance
        rand_ents_1 = np.random.randint(low=0, high=self.n_entities, size=(batchsize*self.neg_ratio,))
        rand_ents_2 = np.random.randint(low=0, high=self.n_entities, size=(batchsize*self.neg_ratio,))

        for i in range(choices.shape[0]):
            r = triples[batchsize+i,1]
            # STRATEGY 0: corrupt either the head or the tail randomly
            if choices[i] == 0:
                if coin_flips[i]: # HEADS! corrupt the head
                    triples[batchsize+i,0] = rand_ents_1[i]
                    # check whether this corruption typechecks, and update the type label
                    if self.joint and rand_ents_1[i] in self.head_dict[r]:
                        values[batchsize+i,1] = 1.0

                else: # TAILS! corrupt the tail
                    triples[batchsize+i,2] = rand_ents_1[i]
                    # check whether this corruption typechecks, and update the type label
                    if self.joint and rand_ents_1[i] in self.tail_dict[r]:
                        values[batchsize+i,1] = 1.0

            # STRATEGY 1: corrupt both the head and the tail
            elif choices[i] == 1:
                # assign the random entities to both head and tail
                triples[batchsize+i,0] = rand_ents_1[i]
                triples[batchsize+i,2] = rand_ents_2[i]
                # check whether this corruption typechecks, and update the type label
                if self.joint and rand_ents_1 in self.head_dict[r] and rand_ents_2 in self.tail_dict[r]:
                    values[batchsize+i, 1] = 1.0

            # STRATEGY 2: corrupt either the head or the tail with something else that
            # typechecks.
            else:
                if coin_flips[i]: # HEADS! corrupt the head
                    self.generate_typed_corr(triples, values, batchsize+i,  self.head_dict_arr, head_or_tail='head')
                else: #TAILS! corrupt the tail
                    self.generate_typed_corr(triples, values, batchsize+i, self.tail_dict_arr, head_or_tail='tail')

        return (t.from_numpy(triples), t.from_numpy(values))

    def generate_typed_corr(self, triples, values, indx, entdict, head_or_tail):
        '''
        Helper function that corrupts the head or tail of triple with the
        given index with type-consistent entities.

        Parameters
        ----------
        triples: numpy.array
            The original triples with shape (batchsize, 3)
        values: numpy.array
            Labels for the original and generated triples
        indx: int
            Index of the triple to be corrupted
        entdict: dict(set) or dict(numpy.array)
            Dictionary with relation ids as keys, and sets of entity ids or binary
            numpy arrays as values. It could be head_dict, tail_dict, head_dict_arr
            or tail_dict_arr, but the latter two is preferable for performance reasons.
        head_or_tail: string
            The position of the entity to be corrupted
            options: 'head', 'tail'
        '''

        assert(entdict is None or type(entdict[0]) == np.ndarray), 'Expected values in the head_dict_arr to be np.ndarray, found %s instead' % str(type(entdict[0]))
        if head_or_tail == 'head':
            loc = 0
        elif head_or_tail == 'tail':
            loc = 2

        rel = int(triples[indx,1])
        # get the array or set of entity indeces that would make the corrupt triple type-consistent
        ents = entdict[rel]

        # pick the entity to corrupt the triple by
        if type(ents) is set:
            raise TypeError('The values of entity dicts needs to be a numpy array! Sets are no longer supported. The type is %s' % str(type(ents)))
#            corr = sample(ents, 1)[0]

        elif type(ents) is np.ndarray:
            corr = np.random.choice(ents)

        else:
            raise TypeError("head and tail dictionaries need to have either sets or numpy arrays as their values.")

        triples[indx, loc] = corr
        values[indx, 1] = 1.0
