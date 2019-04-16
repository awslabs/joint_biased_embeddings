from torch import nn
import torch as t
from .utils import get_correct_and_incorrect_scores, get_device

DEVICE = get_device()

class SoftMaxLoss(nn.modules.loss._Loss):
    '''
    This loss implements NLL of softmax for a batch where each item
    contains one positive, and neg_ratio number of negative samples.

    Attributes
    ----------
    neg_ratio: int
        negatives ratio per positive example
    loss: nn.modules.loss._Loss
        Cross entropy loss
    '''
    def __init__(self, neg_ratio, size_average=True, reduce=True):
        super(SoftMaxLoss, self).__init__(size_average, reduce)
        self.neg_ratio = neg_ratio
        self.loss = t.nn.CrossEntropyLoss()

    def forward(self, scores, _):
        '''
        Function to calculate the loss of a batch. Doesn't use labels to
        infer the correct and incorrect triples, instead relies on the structure of the
        batch: scores[: len(scores)/(neg_ratio + 1)] is assumed to be scores for
        correct triples and the rest to be false triples.

        Parameters
        __________
        scores: torch.FloatTensor
            scores assigned by the model to each triple in the batch

        Returns
        -------
        torch.float
            The multi class cross entropy loss for the batch
        '''
        scores_for_correct_triples, scores_for_incorrect_triples = get_correct_and_incorrect_scores(scores, self.neg_ratio)
        # append the correct scores to the incorrect scores
        scores = t.cat((t.unsqueeze(scores_for_correct_triples, -1), scores_for_incorrect_triples), 1)

        # CrossEntropy expects the correct class for each item as labels.
        # For all triples i in the batch, scores[i,0] is the score of the
        # original triple and scores[i,1:] are scores for the corrupted triples.
        # Hence the correct class is always the 0th.
        labels = t.zeros(scores.shape[0], dtype=t.long).to(DEVICE)

        return self.loss(scores, labels)

class MaxMarginLoss(nn.modules.loss._Loss):
    '''
    This class implements a margin based loss for the batch.

    Attributes
    ----------
    margin: float
        The loss will be 0 if the difference between the scores for positive and
        negative examples is above this level
    neg_ratio: int
        Number of negative examples per positive example
    '''
    def __init__(self, neg_ratio=1, margin=0, size_average=True, reduce=True):
        super(MaxMarginLoss, self).__init__(size_average, reduce)
        self.margin = margin
        self.neg_ratio = neg_ratio
        print("Margin: {}".format(margin))

    def forward(self, scores, _):
        '''
        Function to calculate the loss of a batch. Doesn't use labels to
        infer the correct and incorrect triples, instead relies on the structure of the
        batch.

        Parameters
        __________
        scores: torch.FloatTensor
            scores assigned by the model to each triple in the batch

        Returns
        -------
        torch.float
            Max margin loss for the batch
        '''
        scores_correct, scores_incorrect = get_correct_and_incorrect_scores(scores, self.neg_ratio)
        scores_incorrect = t.transpose(scores_incorrect, 1, 0)
        return t.max(t.Tensor([0]).to(DEVICE), self.margin + scores_incorrect - scores_correct).mean()

class HingeLoss(nn.modules.loss._Loss):
    '''
    This class implements hinge loss for the batch
    '''
    def __init__(self, margin=0, size_average=True, reduce=True):
        super(HingeLoss, self).__init__(size_average, reduce)
        self.margin = margin

    def forward(self, scores, labels):
        # hinge loss expects 1/-1 for labels, so first convert from 1/0
        labels = labels*2 - 1
        return t.max(t.Tensor([0]).to(DEVICE), self.margin - scores*labels).mean()
