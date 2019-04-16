from typing import List, Dict
from torch import nn
from torch.nn.init import xavier_normal_
import torch as t
import abc

from .losses import SoftMaxLoss, MaxMarginLoss, HingeLoss
from .utils import get_device, Parameters
from .experiment import Experiment
from .logging import logger

DEVICE = get_device()

#       _         _                  _                         _       _
#  __ _| |__  ___| |_ _ __ __ _  ___| |_   _ __ ___   ___   __| |_   _| | ___
# / _` | '_ \/ __| __| '__/ _` |/ __| __| | '_ ` _ \ / _ \ / _` | | | | |/ _ \
#| (_| | |_) \__ \ |_| | | (_| | (__| |_  | | | | | | (_) | (_| | |_| | |  __/
# \__,_|_.__/|___/\__|_|  \__,_|\___|\__| |_| |_| |_|\___/ \__,_|\__,_|_|\___|

class Abstract_Module(nn.Module):
    '''Abstract module for all other modules to inherit.

    Attributes
    ----------
    n_entities: int
        number of entities in the dataset
    n_relations: int
        number of relations in the dataset
    embedding_size: int
        number of dimensions of the embeddings to be trained
    loss: torch.nn.modules.loss._Loss
        loss to use during training'''
    def __init__(self, n_entities, n_relations, embedding_size,
                loss, margin=1.0, neg_ratio=1):
        '''
        Parameters
        ----------
        n_entities: int
            number of entities
        n_relations: int
            number of relations
        embedding_size: int
            size of the entity and relation embeddings
        loss: string
            options: "maxmargin", "hinge", "softmax", "softmargin", "full-softmax"
        margin (optional): float
            If the loss is "maxmargin" or "hinge", this will be the margin to use
        neg_ratio (optional): int
            Number of generated negatives per positive example. Needed if the
            loss is "maxmargin" or "softmax"
        '''
        super(Abstract_Module, self).__init__()
        logger.info('Initialising an instance of %s' % self.__class__.__name__)
        self.name = self.__class__.__name__
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_size = embedding_size

        if loss == "maxmargin":
            self.loss = MaxMarginLoss(margin=margin, neg_ratio=neg_ratio)
        elif loss == "hinge":
            self.loss = HingeLoss(margin)
        elif loss == "softmax":
            self.loss = SoftMaxLoss(neg_ratio)
        elif loss == "softmargin":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == "full-softmax":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss passed into module is not recognized.")

    def eval_o(self, e_idx, r_idx):
        '''Evaluates (e, r, _) for all possible entities _.

        This is a wrapper function for eval_o_forward to be used during evaluation,
        where the evaluation script expects a numpy array.

        Parameters
        ----------
        e_idx: torch.LongTensor
            torch array with indexes of the head entities in the batch
        r_idx: torch.LongTensor
            torch array with indexes of the relations in the batch

        Returns
        -------
        numpy.array
            dtype=np.float and shape = (batch_size, n_entities)
            score for triple (h_i, r_i, e_ij) is at result[i, e_ij]'''
        e_idx = t.tensor([e_idx], dtype=t.long).to(DEVICE)
        r_idx = t.tensor([r_idx], dtype=t.long).to(DEVICE)

        with t.no_grad():
            pred = self.eval_o_forward(e_idx, r_idx)
            return pred.data.cpu().numpy()

    def eval_s(self, r_idx, e_idx):
        '''Evaluates (_, r, e) for all possible entities _.

        This is a wrapper function for eval_s_forward to be used during evaluation,
        where the evaluation script expects a numpy array.

        Parameters
        ----------
        r_idx: torch.LongTensor
            torch array with indexes of the relations in the batch
        e_idx: torch.LongTensor
            torch array with indexes of the tail entities in the batch

        Returns
        -------
        numpy.array
            dtype=np.float and shape = (batch_size, n_entities)
            score for triple (e_ij, r_i, h_i) is at result[i, e_ij]'''

        e_idx = t.tensor([e_idx], dtype=t.long).to(DEVICE)
        r_idx = t.tensor([r_idx], dtype=t.long).to(DEVICE)

        with t.no_grad():
            pred = self.eval_s_forward(r_idx, e_idx)
            return pred.data.cpu().numpy()

    # ----- TO BE IMPLEMENTED IN CHILD MODULES ------- #

    @abc.abstractmethod
    def forward(self, e1_idx, r_idx, e2_idx):
        '''Evaluates (e1, r, e2)

        Parameters
        ----------
        e1_idx: torch.LongTensor
            indexes of the head entities
        r_idx: torch.LongTensor
            indexes of the relation
        e2_idx: torch.LongTensor
            indexes of the tail entities

        Returns
        -------
        torch.float
            score assigned by the model for triple.'''
        return

    @abc.abstractmethod
    def eval_o_forward(self, e_idx, r_idx):
        '''Evaluates (e, r, _) for all possible entities _.

        Parameters
        ----------
        e_idx: torch.LongTensor
            indexes of the head entities
        r_idx: torch.LongTensor
            indexes of the relation

        Returns
        -------
        torch.FloatTensor
            shape = (batch_size, n_entities)
            score for triple (h_i, r_i, e_ij) is at result[i, e_ij]'''
        return

    @abc.abstractmethod
    def eval_s_forward(self, r_idx, e_idx):
        '''Evaluates (_, r, e) for all possible entities _.

        Parameters
        ----------
        r_idx: torch.LongTensor
            index of the relations
        e_idx: torch.LongTensor
            index of the tail entities

        Returns
        -------
        torch.FloatTensor
            dtype=np.float and shape = (batch_size, n_entities)
            score for triple (e_ij, r_i, h_i) is at result[i, e_ij]
        '''
        return

    @abc.abstractmethod
    def get_embeddings(self):
        '''Interface for saving the embeddings for the model.
        Returns weights as a tuple of numpy arrays'''
        return

    @abc.abstractmethod
    def get_entity_weights(self):
        '''Interface for tying the weights of entities between type and fact modules.

        The output is intended to be passed to set_embedding_weights. Different
        children modules can return different outputs, but this should match the
        input form for set_embedding_weights'''
        return

    @abc.abstractmethod
    def set_entity_weights(self, weights):
        '''Interface for tying the weights of entities between type and fact modules,
        to be used in combination with get_embedding_weights'''
        return

#  _     _ _ _                                             _       _
# | |__ (_) (_)_ __   ___  __ _ _ __   _ __ ___   ___   __| |_   _| | ___  ___
# | '_ \| | | | '_ \ / _ \/ _` | '__| | '_ ` _ \ / _ \ / _` | | | | |/ _ \/ __|
# | |_) | | | | | | |  __/ (_| | |    | | | | | | (_) | (_| | |_| | |  __/\__ \
# |_.__/|_|_|_|_| |_|\___|\__,_|_|    |_| |_| |_|\___/ \__,_|\__,_|_|\___||___/

class DistMult_Module(Abstract_Module):
    ''' An implementation of the DistMult model described in Yang et al. (2014):
    "Embedding Entities and Relations for Learning and Inference in Knowledge Bases.

    Scoring function is the bilinear product: e_1^T diag(r) e_2"

    Attributes
    ----------
    name: string
    ent_embeddings: torch.nn.Embedding
    rel_embeddings: torch.nn.Embedding
    '''
    def __init__(self, n_entities, n_relations, embedding_size, loss, margin=None, neg_ratio=None):
        # parameters explained in Abstract_Module
        super(DistMult_Module, self).__init__(n_entities, n_relations, embedding_size, loss, margin, neg_ratio)
        self.name = self.__class__.__name__
        self.ent_embeddings = t.nn.Embedding(n_entities, embedding_size)
        # normalize the initialization using Xavier normalization
        xavier_normal_(self.ent_embeddings.weight.data)
        self.rel_embeddings = t.nn.Embedding(n_relations, embedding_size)
        xavier_normal_(self.rel_embeddings.weight.data)


    def eval_o_forward(self, e_idx, r_idx):
        # This is a fast GPU implementation that uses matrix multiplication with the
        # embeddings matrix.

        e1_embed = self.ent_embeddings(e_idx).squeeze()
        rel_embed = self.rel_embeddings(r_idx).squeeze()
        prod = e1_embed*rel_embed
        return t.matmul(prod, self.ent_embeddings.weight.transpose(1,0))

    def eval_s_forward(self, r_idx, e_idx):
        # DistMult is symmetric w.r.t head and tail entities
        return self.eval_o_forward(e_idx, r_idx)

    def forward(self, e1_idx, r_idx, e2_idx):
        score = t.sum(self.ent_embeddings(e1_idx) * self.rel_embeddings(r_idx) * self.ent_embeddings(e2_idx), dim=-1)
        return score

    def get_embeddings(self):
        return (self.ent_embeddings.weight.data.cpu().numpy(), self.rel_embeddings.weight.data.cpu().numpy())

    def get_entity_weights(self):
        return self.ent_embeddings.weight

    def set_entity_weights(self, weights):
        self.ent_embeddings.weight = weights

class Complex_Module(Abstract_Module):
    ''' An implementation of the ComplEx model described in Trouillon et al. (2016):
    complex embeddings for simple link prediction

    Embeddings are in complex space, hence have a real and an imaginary part. The
    scoring function is the real part of the bilinear product: Re(e_1^T diag(r) e_2)

    Following the description of the paper, this is implemented here as:

      Re(e_1)^T diag(Re(r)) Re(e_2)
    + Im(e_1)^T diag(Re(r)) Im(e_2)
    + Re(e_1)^T diag(Im(r)) Im(e_2)
    - Im(e_1)^T diag(Im(r)) Re(e_2)

    Attributes
    ----------
    name: string
    ent_embeddings_r: torch.nn.Embedding
    ent_embeddings_i: torch.nn.Embedding
    rel_embeddings_r: torch.nn.Embedding
    rel_embeddings_i: torch.nn.Embedding
    '''
    def __init__(self, n_entities, n_relations, embedding_size, loss, margin=None, neg_ratio=None):
        # parameters explained in Abstract_Module
        super(Complex_Module, self).__init__(n_entities, n_relations, embedding_size, loss, margin, neg_ratio)

        self.ent_embeddings_r = t.nn.Embedding(n_entities, self.embedding_size)
        self.ent_embeddings_i = t.nn.Embedding(n_entities, self.embedding_size)
        xavier_normal_(self.ent_embeddings_i.weight.data)
        xavier_normal_(self.ent_embeddings_r.weight.data)
        self.rel_embeddings_r = t.nn.Embedding(n_relations, self.embedding_size)
        self.rel_embeddings_i = t.nn.Embedding(n_relations, self.embedding_size)
        xavier_normal_(self.rel_embeddings_i.weight.data)
        xavier_normal_(self.rel_embeddings_r.weight.data)

    def forward(self, e1_idx, r_idx, e2_idx):
        res1 = t.sum(self.ent_embeddings_r(e1_idx) * self.rel_embeddings_r(r_idx) * self.ent_embeddings_r(e2_idx), dim=-1)
        res2 = t.sum(self.ent_embeddings_i(e1_idx) * self.rel_embeddings_r(r_idx) * self.ent_embeddings_i(e2_idx), dim=-1)
        res3 = t.sum(self.ent_embeddings_r(e1_idx) * self.rel_embeddings_i(r_idx) * self.ent_embeddings_i(e2_idx), dim=-1)
        res4 = t.sum(self.ent_embeddings_i(e1_idx) * self.rel_embeddings_i(r_idx) * self.ent_embeddings_r(e2_idx), dim=-1)
        return res1 + res2 + res3 - res4

    def eval_o_forward(self, e_idx, r_idx):
        e1_embed_r = self.ent_embeddings_r(e_idx).squeeze()
        e1_embed_i = self.ent_embeddings_i(e_idx).squeeze()
        rel_embed_r = self.rel_embeddings_r(r_idx).squeeze()
        rel_embed_i = self.rel_embeddings_i(r_idx).squeeze()

        pred1 = t.matmul(e1_embed_r * rel_embed_r, self.ent_embeddings_r.weight.transpose(1,0))
        pred2 = t.matmul(e1_embed_i * rel_embed_r, self.ent_embeddings_i.weight.transpose(1,0))
        pred3 = t.matmul(e1_embed_r * rel_embed_i, self.ent_embeddings_i.weight.transpose(1,0))
        pred4 = t.matmul(e1_embed_i * rel_embed_i, self.ent_embeddings_r.weight.transpose(1,0))

        pred = pred1 + pred2 + pred3 - pred4
        return pred

    def eval_s_forward(self, r_idx, e_idx):

        e2_embed_r = self.ent_embeddings_r(e_idx).squeeze()
        e2_embed_i = self.ent_embeddings_i(e_idx).squeeze()
        rel_embed_r = self.rel_embeddings_r(r_idx).squeeze()
        rel_embed_i = self.rel_embeddings_i(r_idx).squeeze()

        pred1 = t.matmul(rel_embed_r * e2_embed_r, self.ent_embeddings_r.weight.transpose(1,0))
        pred2 = t.matmul(rel_embed_r * e2_embed_i, self.ent_embeddings_i.weight.transpose(1,0))
        pred3 = t.matmul(rel_embed_i * e2_embed_i, self.ent_embeddings_r.weight.transpose(1,0))
        pred4 = t.matmul(rel_embed_i * e2_embed_r, self.ent_embeddings_i.weight.transpose(1,0))

        pred = pred1 + pred2 + pred3 - pred4
        return pred

    def get_entity_weights(self):
        return (self.ent_embeddings_r.weight, self.ent_embeddings_i.weight)

    def set_entity_weights(self, weights):
        self.ent_embeddings_r.weight, self.ent_embeddings_i.weight = weights

    def get_embeddings(self):
        return ((self.ent_embeddings_r.weight.data.cpu().numpy(), self.ent_embeddings_i.weight.data.cpu().numpy()),(self.rel_embeddings_r.weight.data.cpu().numpy(), self.rel_embeddings_i.weight.data.cpu().numpy()))

class SimplE_Module(Abstract_Module):
    ''' An implementation of the SimplE model described in Kazemi et al. (2018):
    SimplE Embedding for Link Prediction in Knowledge Graphs.

    Each entity has two embeddings associated, one for head and one for tail positions.
    Each relation also has two embeddings. One for itself and one for its reverse. Score
    of a triple is calculated by:

    1/2 (e1_head diag(r) e2_tail + e2_head diag(r_rev) e1_tail)

    Attributes
    ----------
    name: string
    ent_embeddings_h: torch.nn.Embedding
    ent_embeddings_t: torch.nn.Embedding
    rel_embeddings: torch.nn.Embedding
    rel_embeddings_rev: torch.nn.Embedding
    '''
    def __init__(self, n_entities, n_relations, embedding_size, loss, margin=None, neg_ratio=None):

        super(SimplE_Module, self).__init__(n_entities, n_relations, embedding_size, loss, margin, neg_ratio)

        self.ent_embeddings_h = t.nn.Embedding(n_entities, self.embedding_size)
        self.ent_embeddings_t = t.nn.Embedding(n_entities, self.embedding_size)
        xavier_normal_(self.ent_embeddings_h.weight.data)
        xavier_normal_(self.ent_embeddings_t.weight.data)
        self.rel_embeddings = t.nn.Embedding(n_relations, self.embedding_size)
        self.rel_embeddings_rev = t.nn.Embedding(n_relations, self.embedding_size)
        xavier_normal_(self.rel_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings_rev.weight.data)

    def forward(self, e1_idx, r_idx, e2_idx):
        res1 = t.sum(self.ent_embeddings_h(e1_idx) * self.rel_embeddings(r_idx) * self.ent_embeddings_t(e2_idx), dim=-1)
        res2 = t.sum(self.ent_embeddings_t(e1_idx) * self.rel_embeddings_rev(r_idx) * self.ent_embeddings_h(e2_idx), dim=-1)
        return (res1 + res2)/2

    def eval_o_forward(self, e_idx, r_idx):

        e1_embed_h = self.ent_embeddings_h(e_idx).squeeze()
        e1_embed_t = self.ent_embeddings_t(e_idx).squeeze()
        rel_embed = self.rel_embeddings(r_idx).squeeze()
        rel_embed_rev = self.rel_embeddings_rev(r_idx).squeeze()

        pred1 = t.matmul(e1_embed_h * rel_embed, self.ent_embeddings_t.weight.transpose(1,0))
        pred2 = t.matmul(e1_embed_t * rel_embed_rev, self.ent_embeddings_h.weight.transpose(1,0))

        pred = (pred1 + pred2)/2
        return pred

    def eval_s_forward(self, r_idx, e_idx):

        e2_embed_h = self.ent_embeddings_h(e_idx).squeeze()
        e2_embed_t = self.ent_embeddings_t(e_idx).squeeze()
        rel_embed = self.rel_embeddings(r_idx).squeeze()
        rel_embed_rev = self.rel_embeddings_rev(r_idx).squeeze()

        pred1 = t.matmul( rel_embed * e2_embed_t, self.ent_embeddings_h.weight.transpose(1,0))
        pred2 = t.matmul( rel_embed_rev * e2_embed_h, self.ent_embeddings_t.weight.transpose(1,0))

        pred = (pred1 + pred2)/2
        return pred

    def get_entity_weights(self):
        return (self.ent_embeddings_h.weight, self.ent_embeddings_t.weight)

    def set_entity_weights(self, weights):
        self.ent_embeddings_h.weight, self.ent_embeddings_t.weight = weights

    def get_embeddings(self):
        return ((self.ent_embeddings_h.weight.data.cpu().numpy(), self.ent_embeddings_t.weight.data.cpu().numpy()),(self.rel_embeddings.weight.data.cpu().numpy(), self.rel_embeddings_rev.weight.data.cpu().numpy()))

#  _                       _       _   _                   _                       _       _
# | |_ _ __ __ _ _ __  ___| | __ _| |_(_) ___  _ __   __ _| |  _ __ ___   ___   __| |_   _| | ___  ___
# | __| '__/ _` | '_ \/ __| |/ _` | __| |/ _ \| '_ \ / _` | | | '_ ` _ \ / _ \ / _` | | | | |/ _ \/ __|
# | |_| | | (_| | | | \__ \ | (_| | |_| | (_) | | | | (_| | | | | | | | | (_) | (_| | |_| | |  __/\__ \
#  \__|_|  \__,_|_| |_|___/_|\__,_|\__|_|\___/|_| |_|\__,_|_| |_| |_| |_|\___/ \__,_|\__,_|_|\___||___/

class TransE_Module(Abstract_Module):
    '''Abstract implementation of the TransE model as described in Bordes et al. (2013):
    "Translating embeddings for modeling multi-relational data. Child classes need to
    implement which norm to use"

    The scoring function is: - L_p(e_1 + r - e_2)

    Attributes
    ----------
    ent_embeddings: torch.nn.Embedding
    rel_embeddings: torch.nn.Embedding
    '''
    def __init__(self, n_entities, n_relations, embedding_size, loss, margin=1.0, neg_ratio=1):

        super(TransE_Module, self).__init__(n_entities, n_relations, embedding_size, loss, margin=margin, neg_ratio=neg_ratio)

        self.ent_embeddings = t.nn.Embedding(n_entities, embedding_size)
        xavier_normal_(self.ent_embeddings.weight.data)
        self.rel_embeddings = t.nn.Embedding(n_relations, embedding_size)
        xavier_normal_(self.rel_embeddings.weight.data)

    def get_entity_weights(self):
        return self.ent_embeddings.weight

    def set_entity_weights(self, weights):
        self.ent_embeddings.weight = weights

    def get_embeddings(self):
        return (self.ent_embeddings.weight.data.cpu().numpy(), self.rel_embeddings.weight.data.cpu().numpy())
    ### needs to be implemented in the children modules ###

    @abc.abstractmethod
    def forward(self,  e1_idx, r_idx, e2_idx):
        return

    @abc.abstractmethod
    def eval_o_forward(self, e_idx, r_idx):
        return

    @abc.abstractmethod
    def eval_s_forward(self, r_idx, e_idx):
        return

class TransE_L1_Module(TransE_Module):
    ''' Implementation of the TransE model using L1 norm'''
    def __init__(self, n_entities, n_relations, embedding_size, loss, margin=1.0, neg_ratio=1):
        super(TransE_L1_Module, self).__init__(n_entities, n_relations, embedding_size, loss, margin=margin, neg_ratio=neg_ratio)

    def forward(self, e1_idx, r_idx, e2_idx):
        return -t.sum(t.abs(self.ent_embeddings(e1_idx) + self.rel_embeddings(r_idx) - self.ent_embeddings(e2_idx)), dim=-1)

    def eval_o_forward(self, e_idx, r_idx):
        # When batch size is 1 e_idx.shape returns ()
        try:
            batch_size = e_idx.shape[0]
        except IndexError:
            batch_size = 1

        e1_embed = self.ent_embeddings(e_idx).view(batch_size, 1, self.embedding_size)
        rel_embed = self.rel_embeddings(r_idx).view(batch_size, 1, self.embedding_size)
        ent_embeddings = self.ent_embeddings.weight.view(1, self.n_entities, self.embedding_size)
        pred = -t.sum(t.abs(e1_embed + rel_embed - ent_embeddings), 2)
        return pred.squeeze()

    def eval_s_forward(self, r_idx, e_idx):
        try:
            batch_size = e_idx.shape[0]
        except IndexError:
            batch_size = 1

        e2_embed = self.ent_embeddings(e_idx).view(batch_size, 1, self.embedding_size)
        rel_embed = self.rel_embeddings(r_idx).view(batch_size, 1, self.embedding_size)
        ent_embeddings = self.ent_embeddings.weight.view(1, self.n_entities, self.embedding_size)

        pred = -t.sum(t.abs(ent_embeddings + rel_embed - e2_embed), 2)
        return pred.squeeze()

class TransE_L2_Module(TransE_Module):
    ''' Implementation of the TransE model using L2 norm '''
    def __init__(self, n_entities, n_relations, embedding_size, loss, margin=None, neg_ratio=None):
        super(TransE_L2_Module, self).__init__(n_entities, n_relations, embedding_size, loss, margin, neg_ratio)

    def forward(self, e1_idx, r_idx, e2_idx):
        return -t.norm(self.ent_embeddings(e1_idx) + self.rel_embeddings(r_idx) - self.ent_embeddings(e2_idx), dim=-1)

    def eval_o_forward(self, e_idx, r_idx):
        e1_embed = self.ent_embeddings(e_idx).squeeze()
        rel_embed = self.rel_embeddings(r_idx).squeeze()
        pred = -t.sum((e1_embed + rel_embed - self.ent_embeddings.weight).pow(2), 1)
        return pred

    def eval_s_forward(self, r_idx, e_idx):
        e2_embed = self.ent_embeddings(e_idx).squeeze()
        rel_embed = self.rel_embeddings(r_idx).squeeze()
        pred = -t.sum((self.ent_embeddings.weight + rel_embed - e2_embed).pow(2), 1)
        return pred

 #   _       _       _                         _       _
 #  (_) ___ (_)_ __ | |_   _ __ ___   ___   __| |_   _| | ___
 #  | |/ _ \| | '_ \| __| | '_ ` _ \ / _ \ / _` | | | | |/ _ \
 #  | | (_) | | | | | |_  | | | | | | (_) | (_| | |_| | |  __/
 # _/ |\___/|_|_| |_|\__| |_| |_| |_|\___/ \__,_|\__,_|_|\___|
# |__/

class Joint_Module(nn.Module):
    '''Module for joint training. TODO: add the paper link.

    Attributes
    ----------
    fact_module: Abstract_Module
        Initialized module to use for the training on the correctness of the triples
    type_module: Abstract_Module
        Initialized module to use for the training on whether the triple type-checks
    loss_ratio: float
        The value to weight the loss contributed by the type module
    n_entities: int
        Number of entities
    n_relations: int
        Number of relations
    embedding_size: int
        Size of the entity and relation embeddings'''

    def __init__(self, fact_module, type_module, loss_ratio):
        '''
        Parameters
        ----------
        fact_module: Abstract_Module
        type_module: Abstract_Module
        loss_ratio: float
        '''
        # confirm that the dimensions for the embeddings match between the two modules
        assert fact_module.n_entities == type_module.n_entities, "Number of entities in fact module and type module are different"
        assert fact_module.n_relations == type_module.n_relations, "Number of relations in fact module and type module are different"
        assert fact_module.embedding_size == type_module.embedding_size, "Embedding size in the fact modula and type module are different"

        super(Joint_Module, self).__init__()
        self.name = self.__class__.__name__
        self.fact_module = fact_module
        self.type_module = type_module

        # tie the entity weights between the type and the fact module
        self.type_module.set_entity_weights(self.fact_module.get_entity_weights())

        self.loss_ratio = loss_ratio
        self.n_entities = fact_module.n_entities
        self.n_relations = fact_module.n_relations
        self.embedding_size = fact_module.embedding_size

    def loss(self, preds, labels):
        ''' Calculates the loss for the joint model given the predictions and labels.

        Parameters
        ----------
        preds: (torch.FloatTensor, torch.FloatTensor)
            predicted values from the forward pass of the model from the fact module
            and the type module respectively.
        labels: (torch.FloatTensor, torch.FloatTensor) or (torch.LongTensor, torch.FloatTensor)
            True labels for the type module and the fact module respectively. If the
            fact loss is full softmax, then the first index contains the indeces of the correct
            entities.

        Returns
        -------
        (total_loss:torch.float , fact_loss:torch.float , type_loss:torch.float)
            total_loss is the overall loss, fact_loss is the loss contributed by the
            fact module and type_loss is the (non-scaled) loss from the type module
        '''
        ans1, ans2 = labels
        pred1, pred2 = preds
        loss1 = self.fact_module.loss(pred1, ans1)
        loss2 = self.type_module.loss(pred2, ans2)
        return (loss1 + self.loss_ratio * loss2, loss1, loss2)

    def joint_loss_with_full_softmax(self, scores_f, scores_t, ent_idx, type_labels):
        loss1 = self.fact_module.loss(scores_f, ent_idx)
        loss2 = self.type_module.loss(scores_t, type_labels)
        return (loss1 + self.loss_ratio * loss2, loss1, loss2)

    def forward(self, e1_idx, r_idx, e2_idx):
        '''Evaluates (e1, r, e2)

        Parameters
        ----------
        e1_idx: torch.LongTensor
            indexes of the head entities
        r_idx: torch.LongTensor
            indexes of the relation
        e2_idx: torch.LongTensor
            indexes of the tail entities

        Returns
        -------
        (torch.float, torch.float)
            score assigned to the triple by the fact module and by the type module '''
        ans1 = self.fact_module.forward(e1_idx, r_idx, e2_idx)
        ans2 = self.type_module.forward(e1_idx, r_idx, e2_idx)
        return (ans1, ans2)

    def eval_o_forward(self, e_idx, r_idx):
        ''' Evaluates (e, r, _) for all entities with the fact module'''
        return self.fact_module.eval_o_forward(e_idx, r_idx)

    def eval_s_forward(self, r_idx, e_idx):
        ''' Evaluates (_, r, e) for all entities with the fact module'''
        return self.fact_module.eval_s_forward(r_idx, e_idx)

    def eval_o_forward_type(self, e_idx, r_idx):
        ''' Evaluates (e, r, _) for all entities with the type module'''
        return self.type_module.eval_o_forward(e_idx, r_idx)

    def eval_s_forward_type(self, r_idx, e_idx):
        ''' Evaluates (_, r, e) for all entities with the type module'''
        return self.type_module.eval_s_forward(r_idx, e_idx)

    def eval_o(self, e_idx, r_idx):
        ''' Wrapper that evalues (e, r, _) and returns the result as a numpy array '''
        e_idx = t.tensor([e_idx], dtype=t.long).to(DEVICE)
        r_idx = t.tensor([r_idx], dtype=t.long).to(DEVICE)

        with t.no_grad():
            pred = self.eval_o_forward(e_idx, r_idx)
            return pred.data.cpu().numpy()

    def eval_s(self, r_idx, e_idx):
        ''' Wrapper that evalues (_, r, e) and returns the result as a numpy array '''
        e_idx = t.tensor([e_idx], dtype=t.long).to(DEVICE)
        r_idx = t.tensor([r_idx], dtype=t.long).to(DEVICE)

        with t.no_grad():
            pred = self.eval_s_forward(r_idx, e_idx)
            return pred.data.cpu().numpy()

    def get_embeddings(self):
        '''Get embeddings for the fact module'''
        return self.fact_module.get_embeddings()


def create_module(experiment: Experiment, method: str, joint: bool, parameters: Parameters):
    module_cls = MODULES[method]
    if joint:
        module_1 = module_cls(experiment.n_entities, experiment.n_relations,
                                        loss= parameters.fact_loss, embedding_size= parameters.embedding_size,
                                        neg_ratio = parameters.neg_ratio)
        module_2 = module_cls(experiment.n_entities, experiment.n_relations,
                                        loss= parameters.type_loss, embedding_size= parameters.embedding_size,
                                        neg_ratio = parameters.neg_ratio)
        module = Joint_Module(module_1, module_2, parameters.loss_ratio)
    else:
        module = module_cls(experiment.n_entities, experiment.n_relations,
                                        loss= parameters.fact_loss, embedding_size= parameters.embedding_size,
                                        neg_ratio = parameters.neg_ratio)

    return module


MODULES = {'complex': Complex_Module, 'distmult': DistMult_Module, 'simple': SimplE_Module, 
           'transe_l1': TransE_L1_Module, 'transe_l2': TransE_L2_Module}
BILINEAR_MODULES = {'complex': Complex_Module, 'distmult': DistMult_Module, 'simple': SimplE_Module}

