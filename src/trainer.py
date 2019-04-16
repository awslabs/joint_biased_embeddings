import torch as t
from torch import nn
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.utils import data
from collections import defaultdict
import numpy as np

from time import time
from statistics import mean
import subprocess

from .logging import *
from .experiment import Experiment
from .batching import TriplesDataset, TriplesDatasetTyped, TriplesDatasetFullSoftmax
from .modules import DistMult_Module, Complex_Module, SimplE_Module, TransE_L1_Module, TransE_L2_Module, Joint_Module
from .losses import SoftMaxLoss, MaxMarginLoss, HingeLoss
from .utils import get_device, log_results, Parameters

DEVICE = get_device()

def worker_init_fn(worker_id):
    '''helper class for initializing the random seed in worker processes. Allows batching to
    be random if the multiprocessing functionality of the batcher is used. To be
    passed into the initializer for the subprocess.'''
    np.random.seed((t.randint(low=0, high=2**32 - 1, size=(1,), dtype=t.int64) + worker_id)%2**32 - 1)

class Trainer(object):
    '''
    Trainer class to encapsulate the training of a given module.

    Attributes
    ----------
    module: Abstract_Module
        the module to do the training, validation and testing on
    '''
    def __init__(self, module):
        self.module = module
        self.module.to(DEVICE)

    def constrain(self):
        '''Renormalizes the embeddings of the module so that they are each length 1.
        To be used with max margin loss.'''
        with t.no_grad():
            for param in self.module.parameters():
                param.data.renorm_(2, 0, 1)

    def fit(self, experiment, parameters):
        # TODO: it would be neater to take print_test as an argument here rather
        # than a part of the parameters.
        '''
        Train the module with the given data and parameters.

        Parameters
        ----------
        experiment: Experiment
            object encapsulating the dataset. see experiment.py for details
        parameters: Parameters
            object encapsulating the hyperparameters. see utils.py for details

        Returns
        -------
        dict
            a dictionary containing the keys 'hits@1', 'hits@3', 'hits@10', 'mrr' and 'rmrr'
        '''
        logger.info("Training...")
        # set the learning rate policy
        if parameters.policy == 'adagrad':
            optimizer = t.optim.Adagrad(self.module.parameters(), lr=parameters.learning_rate, weight_decay=parameters.lmbda)

        elif parameters.policy =='adam':
            optimizer = t.optim.Adam(self.module.parameters(), lr=parameters.learning_rate, weight_decay=parameters.lmbda)

        else:
            raise ValueError("learning rate policy should either be 'adam' or 'adagrad'")

        if isinstance(self.module, Joint_Module) :
            joint = True
        else:
            joint = False

        best_valid = {'hits@1': -1.0, 'hits@3': -1.0, 'hits@10': -1.0, 'mrr': -1.0, 'rmrr': -1.0}
        iters = 0
        epoch_times = []

        for epoch in range(parameters.max_iters):

            start_t = time()
            iters = self.run_epoch(joint, iters, parameters, experiment, optimizer)
            time_passed = time() - start_t
            epoch_times.append(time_passed)

            if epoch % parameters.valid_interval == 0:
                logger.info("Average time per epoch is {}".format(mean(epoch_times)))
                epoch_times = []
                valid_results, _, _ = self.score(experiment, experiment.valid_triples)
                logger.info("Performance on validation set:")
                log_results(valid_results)

                if best_valid[parameters.early_stop_with] > valid_results[parameters.early_stop_with]:
                    logger.info("validation {} decreased, stopping here".format(parameters.early_stop_with))
                    break

                else:
                    best_valid = valid_results
                    logger.info("Updating the checkpointed model...")
                    if parameters.checkpoint_file:
                        t.save(self.module.state_dict(), parameters.checkpoint_file)

        if parameters.print_test:

            if parameters.checkpoint_file:
                # load from the last checkpoint where the model had performed better
                self.module.load_state_dict(t.load(parameters.checkpoint_file))
            test_results_overall, test_results_obj, test_results_subj = self.score(experiment, experiment.test_triples)
            logger.info("Performance on test set:")
            log_results(test_results_overall)
            logger.info("Performance on objects")
            log_results(test_results_obj)
            logger.info("Performance on subjects")
            log_results(test_results_subj)

            return test_results_overall

        return best_valid

    def test(self, experiment: Experiment, checkpoint_file):
        self.module.load_state_dict(t.load(checkpoint_file))
        test_results_overall, test_results_obj, test_results_subj = self.score(experiment, experiment.test_triples)
        logger.info("Performance on test set:")
        log_results(test_results_overall)
        logger.info("Performance on objects")
        log_results(test_results_obj)
        logger.info("Performance on subjects")
        log_results(test_results_subj)

        return test_results_overall

    def score(self, experiment, triples):
        '''
        Evaluate the model on the given triples.

        Parameters:
        ----------
        experiment: Experiment
            object encapsulating the dataset. see experiment.py for details
        triples: iterator(tuple)
            list/iterator where each element is a triple in the set to be scored
        Returns
        -------
        dict
            a dictionary with hits@1, 3, 10, mrr and rmrr
        '''
        raw_ranks_obj = []
        raw_ranks_subj = []
        filt_ranks_obj = []
        filt_ranks_subj = []

        for head_idx, rel_idx, tail_idx in triples:
            # Do an all-at-once evaluation for the tail position
            object_results = self.module.eval_o(head_idx, rel_idx)

            # rank of the triple is equal to the number of entities that it scored lower than, plus one.
            r_rank_o = np.sum(object_results > object_results[tail_idx]) + 1
            raw_ranks_obj.append(r_rank_o)
            # get the scores for the objects to be filtered
            known_results = object_results[experiment.known_obj_triples[(head_idx, rel_idx)]]
            to_filter = np.sum(known_results > object_results[tail_idx])
            # subtract these from the raw counts
            filt_ranks_obj.append(r_rank_o - to_filter)

            #Calculate the same for the subjects
            subject_results = self.module.eval_s(rel_idx, tail_idx)

            # rank of the a'th triple is equal to the number of entities that it scored lower than, plus one.
            r_rank_s = np.sum(subject_results > subject_results[head_idx]) + 1
            raw_ranks_subj.append(r_rank_s)
            # get the scores for the subjects to be filtered
            known_results = subject_results[experiment.known_subj_triples[(rel_idx, tail_idx)]]
            to_filter = np.sum(known_results > subject_results[head_idx])
            # subtract these from the raw counts
            filt_ranks_subj.append(r_rank_s - to_filter)

        raw_ranks_obj = np.asarray(raw_ranks_obj)
        raw_ranks_subj = np.asarray(raw_ranks_subj)
        filt_ranks_obj = np.asarray(filt_ranks_obj)
        filt_ranks_subj = np.asarray(filt_ranks_subj)

        subj_result = {'mrr': np.mean(1/filt_ranks_subj), 'rmrr': np.mean(1/raw_ranks_subj)}
        obj_result = {'mrr': np.mean(1/filt_ranks_obj), 'rmrr': np.mean(1/raw_ranks_obj)}

        subj_result['hits@1'] = np.sum(filt_ranks_subj <= 1)/filt_ranks_subj.shape[0]
        obj_result['hits@1'] = np.sum(filt_ranks_obj <= 1)/filt_ranks_obj.shape[0]

        subj_result['hits@3'] = np.sum(filt_ranks_subj <= 3)/filt_ranks_subj.shape[0]
        obj_result['hits@3'] = np.sum(filt_ranks_obj <= 3)/filt_ranks_obj.shape[0]

        subj_result['hits@10'] = np.sum(filt_ranks_subj <= 10)/filt_ranks_subj.shape[0]
        obj_result['hits@10'] = np.sum(filt_ranks_obj <= 10)/filt_ranks_obj.shape[0]

        overall_result = {key: (subj_result[key] + obj_result[key])/2 for key in subj_result}

        return overall_result, obj_result, subj_result

    def save_top_preds(self, experiment, filename):
        '''
        Save the top ten predictions for the head and the tail for each triple
        in the validation set.

        Parameters
        ----------
        experiment: Experiment
            object encapsulating the dataset. see experiment.py for details
        filename: string
            name of the file to save the predictions to
        '''
        logger.info("saving predictions to file...")

        preds_dict = defaultdict(dict)
        for (i,j,k) in experiment.valid_triples:
            #Computing objects ranks
            known_obj_triples = [item for item in experiment.known_obj_triples[(i,j)] if item != k]
            known_subj_triples = [item for item in experiment.known_sub_triples[(j,k)] if item != i]
            # score agains all objects
            scores_obj = self.module.eval_o(i,j)
            scores_subj = self.module.eval_s(j,k)
            # sort, get the raw ranking
            ranks_obj = list(np.argsort(-scores_obj, axis=None))
            ranks_subj = list(np.argsort(-scores_subj, axis=None))
            # filter out the known objects
            ranks_obj = [idx for idx in ranks_obj if idx not in known_obj_triples]
            ranks_subj = [idx for idx in ranks_subj if idx not in known_subj_triples]

            top_ten_obj = ranks_obj[:10]
            top_ten_subj = ranks_subj[:10]

            preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj'] = [ent_dict[idx] for idx in top_ten_obj]
            preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj'] = [ent_dict[idx] for idx in top_ten_subj]

            if k in top_ten_obj:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj@10'] = True
            else:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj@10'] = False
            if k in top_ten_obj[:3]:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj@3'] = True
            else:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj@3'] = False
            if k in top_ten_obj[:1]:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj@1'] = True
            else:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['obj@1'] = False

            if i in top_ten_subj:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj@10'] = True
            else:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj@10'] = False
            if i in top_ten_subj[:3]:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj@3'] = True
            else:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj@3'] = False
            if i in top_ten_subj[:1]:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj@1'] = True
            else:
                preds_dict[(ent_dict[i], rel_dict[j], ent_dict[k])]['subj@1'] = False

        pickle.dump(preds_dict, open(filename, "wb"))


    def run_epoch(self, joint, num_iters, parameters, experiment, optimizer, mixed_sample=True):
        #TODO: separate the ifs into individual functions
        '''
        Run the training for one epoch.

        Parameters
        ----------
        joint: bool
            True if the training is done jointly, False otherwise
        num_iters: int
            number of batches that the training has gone through so far. Used
            in determining when to print out logging for loss.
        experiment: Experiment
            object encapsulating the dataset. see experiment.py for details
        parameters: Parameters
            object encapsulating the hyperparameters. see utils.py for details
        optimizer: torch.optim.Optimizer

        Returns
        -------
        int
            updated number of batches the training has gone through
        '''
        if parameters.fact_loss == 'full-softmax' and joint:
            # training procedure for full softmax and joint training.
            criterion = self.module.joint_loss_with_full_softmax
            traindata = TriplesDatasetFullSoftmax(experiment.train_triples,
                                n_entities = experiment.n_entities,
                                batch_size = parameters.batch_size,
                                head_dict_arr= experiment.head_dict_arr,
                                tail_dict_arr= experiment.tail_dict_arr,
                                joint = True
                                )
            train = data.DataLoader(traindata, **parameters.batch_params, worker_init_fn=worker_init_fn, shuffle=False)

            for triples, labels in train:

                optimizer.zero_grad()
                triples = triples.view(-1, 3)

                if triples.shape[0] == 1: # then errors appear in downstream code due to use of .squeeze(), so discard batch
                    continue

                e1, r, e2 = triples[:,0].to(DEVICE), triples[:,1].to(DEVICE), triples[:,2].to(DEVICE)

                #ensure the labels are in the correct shape, and put them on the GPU
                type_labels_head_corr = labels[0].view(triples.size()[0], experiment.n_entities).to(DEVICE)
                type_labels_tail_corr = labels[1].view(triples.size()[0], experiment.n_entities).to(DEVICE)

                #get the head and tail scores for the fact module, reshape
                scores_tail = self.module.eval_o_forward(e1, r).view(e1.shape[0], experiment.n_entities)
                scores_head = self.module.eval_s_forward(r, e2).view(e1.shape[0], experiment.n_entities)

                # get the head and the tail scores for the type module, reshape
                scores_tail_type = self.module.eval_o_forward_type(e1, r).view(e1.shape[0], experiment.n_entities)
                scores_head_type = self.module.eval_s_forward_type(r, e2).view(e1.shape[0], experiment.n_entities)


                loss1 = criterion(scores_head, scores_head_type, e1, type_labels_head_corr)
                loss2 = criterion(scores_tail, scores_tail_type, e2, type_labels_tail_corr)
                # overall loss to backprop
                loss = loss1[0] + loss2[0]

                # individual losses for logging
                fact_loss = loss1[1] + loss2[1]
                type_loss = loss1[2] + loss2[2]

                if t.isnan(loss):
                    logger.info("The loss has become NAN!")
                    return 0.0

                loss.backward()
                optimizer.step()

                if isinstance(self.module.loss, MaxMarginLoss):
                    self.constrain()

                num_iters += 1
                if num_iters % 100 == 0:
                    logger.info("batch number {} \t total loss: {} \t fact loss: {} \t type loss: {}".format(num_iters, loss, fact_loss, type_loss))

            return num_iters

        elif parameters.fact_loss == 'full-softmax' and not joint:
            # training procedure for full-softmax only, without the joint training
            criterion = self.module.loss
            traindata = TriplesDatasetFullSoftmax(experiment.train_triples,
                                n_entities = experiment.n_entities,
                                batch_size = parameters.batch_size,
                                head_dict_arr = experiment.head_dict_arr,
                                tail_dict_arr = experiment.tail_dict_arr,
                                joint = False
                                )
            train = data.DataLoader(traindata, **parameters.batch_params, worker_init_fn=worker_init_fn, shuffle=False)

            for triples in train:

                optimizer.zero_grad()
                triples = triples.view(-1, 3)

                if triples.shape[0] == 1: # then errors appear in downstream code due to use of .squeeze(), so discard batch
                    continue

                e1, r, e2 = triples[:,0].to(DEVICE), triples[:,1].to(DEVICE), triples[:,2].to(DEVICE)

                scores_tail = self.module.eval_o_forward(e1, r).view(e1.shape[0], experiment.n_entities)
                scores_head = self.module.eval_s_forward(r, e2).view(e1.shape[0], experiment.n_entities)

                loss = criterion(scores_head, e1) + criterion(scores_tail, e2)

                if t.isnan(loss):
                    logger.info("The loss has become NAN!")
                    return 0.0

                loss.backward()
                optimizer.step()

                num_iters += 1
                if num_iters % 100 == 0:
                    logger.info("batch number {} \t loss: {}".format(num_iters, loss))

            return num_iters

        elif mixed_sample and joint:
            # training procedure for joint training where the negative samples are
            # generated by corrupting the head and the tail in a mixed, random fashion.
            criterion = self.module.loss

            assert(experiment.head_dict_arr is None or type(experiment.head_dict_arr[0]) == np.ndarray), 'Expected values in the head_dict_arr to be np.ndarray, found %s instead' % str(type(experiment.head_dict_arr[0]))
            traindata = TriplesDatasetTyped(experiment.train_triples,
                                n_entities = experiment.n_entities,
                                batch_size = parameters.batch_size,
                                neg_ratio = parameters.neg_ratio,
                                head_dict = experiment.head_dict,
                                tail_dict = experiment.tail_dict,
                                head_dict_arr = experiment.head_dict_arr,
                                tail_dict_arr = experiment.tail_dict_arr,
                                type_ratios = parameters.type_ratios,
                                )
            train = data.DataLoader(traindata, **parameters.batch_params, worker_init_fn=worker_init_fn, shuffle=False)

            for triples, labels in train:


                optimizer.zero_grad()
                triples = triples.view(-1, 3)
                labels = labels.view(triples.size()[0], -1)

                if triples.shape[0] == 1:
                    continue

                e1, r, e2 = triples[:,0].to(DEVICE), triples[:,1].to(DEVICE), triples[:,2].to(DEVICE)
                label = (labels[:,0].to(DEVICE), labels[:,1].to(DEVICE))

                scores = self.module.forward(e1, r, e2)
                loss = criterion(scores, label)
                loss, fact_loss, type_loss = loss

                if t.isnan(loss):
                    logger.info("The loss has become NAN!")
                    return 0.0

                loss.backward()
                optimizer.step()

                if isinstance(self.module.loss, MaxMarginLoss):
                    self.constrain()

                num_iters += 1
                if num_iters % 100 == 0:
                    logger.info("batch number {} \t total loss: {} \t fact loss: {} \t type loss: {}".format(num_iters, loss, fact_loss, type_loss))

            return num_iters

        elif mixed_sample and not joint:
            # training procedure for regular training with mixed negative sampling
            criterion = self.module.loss
            traindata = TriplesDataset(experiment.train_triples,
                                        batch_size = parameters.batch_size,
                                        n_entities = experiment.n_entities,
                                        neg_ratio = parameters.neg_ratio,
                                        )

            train = data.DataLoader(traindata, **parameters.batch_params, worker_init_fn=worker_init_fn, shuffle=False)

            for triples, labels in train:
                optimizer.zero_grad()
                triples = triples.view(-1, 3)
                labels = labels.view(triples.size()[0],)
                e1, r, e2 = triples[:,0].to(DEVICE), triples[:,1].to(DEVICE), triples[:,2].to(DEVICE)
                label = labels.to(DEVICE)

                scores = self.module.forward(e1, r, e2)
                loss = criterion(scores, label)

                if t.isnan(loss):
                    logger.info("The loss has become NAN!")
                    return 0.0

                loss.backward()
                optimizer.step()

                if isinstance(self.module.loss, MaxMarginLoss):
                    self.constrain()

                num_iters += 1
                if num_iters % 100 == 0:
                    logger.info("batch number {} \t loss: {}".format(num_iters, loss))

            return num_iters

        else:
            # TODO: implement the versions for seperate sampled negatives,
            # where each triple has two associated sets: one with only corrupted
            # head entities and one with only corrupted tail entities
            raise NotImplementedError("Batches where only the head or the tail is corrupted is not implemented yet.")


    def predict(self, e1_idx, r_idx, e2_idx):
        '''
        Predict the score of the triple for the model.
        This is a wrapper for self.module.forward

        Parameters
        ----------
        e1_idx: list or numpy.array
        e2_idx: list or numpy.array
        r_idx: list or numpy.array

        Returns
        -------
        numpy.array
            scores for the triples

        Notes
        -----
        length of the three arguments must be the same

        '''
        with t.no_grad():
            e1_idx = t.from_numpy(np.asarray(e1_idx)).to(DEVICE)
            r_idx = t.from_numpy(np.asarray(r_idx)).to(DEVICE)
            e2_idx = t.from_numpy(np.asarray(e2_idx)).to(DEVICE)
            preds = self.module.forward(e1_idx, r_idx, e2_idx)
            if self.module is Joint_Module:
                return preds[0].data.cpu().numpy()
            else:
                return preds.data.cpu().numpy()

    def eval_o(self, e_idx, r_idx):
        '''
        Perform all at once evaluation for the tail entity.
        This is a wrapper function for self.module.eval_o

        Parameters
        ----------
        e_idx: list or numpy.array
        r_idx: list or numpy.array

        Returns
        -------
        numpy.array
        '''
        e_idx = t.from_numpy(np.asarray(e_idx)).to(DEVICE)
        r_idx = t.from_numpy(np.asarray(r_idx)).to(DEVICE)

        return self.module.eval_o(e_idx, r_idx)

    def eval_s(self, r_idx, e_idx):
        '''
        Perform all at once evaluation for the head entity.
        This is a wrapper function for self.module.eval_s

        Parameters
        ----------
        e_idx: list or numpy.array
        r_idx: list or numpy.array

        Returns
        -------
        numpy.array
        '''
        e_idx = t.from_numpy(np.asarray(e_idx)).to(DEVICE)
        r_idx = t.from_numpy(np.asarray(r_idx)).to(DEVICE)
        return self.module.eval_s(r_idx, e_idx)

    def get_embeddings(self):
        '''
        Get embeddings of the module. Wrapper for self.module.get_embeddings

        Returns
        -------
        numpy.array tuple
        '''
        return self.module.get_embeddings()
