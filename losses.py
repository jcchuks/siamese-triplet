import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class OnlineTripletLossV2(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLossV2, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()
        losses = F.relu(ap_distances - an_distances + self.margin)
        return losses.mean(), len(triplets)

class OnlineTripletLossV3(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    Balntas et al
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLossV3, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        A = 0
        P = 1
        N = 2
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
	
        anchor = embeddings[triplets[:, A]]
        positives = embeddings[triplets[:, P]]
        negatives = embeddings[triplets[:, N]]
        ap_distances = torch.diagonal(torch.cdist(anchor, positives))
        an_distances = torch.diagonal(torch.cdist(anchor, negatives))
        pn_distances = torch.diagonal(torch.cdist(positives, negatives))
        an_pn = torch.minimum(an_distances, pn_distances)
        losses = F.relu(ap_distances - an_pn + self.margin)

        #loss_func = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), 
        #					     swap=True, margin=0.9)
        #losses = loss_func(embeddings[triplets[:, A]], embeddings[triplets[:, P]], embeddings[triplets[:, N]])

        return losses.mean(), len(triplets)


class OnlineTripletLossV4(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    Zhang et al
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLossV4, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        A = 0
        P = 1
        N = 2
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
	
        anchor = embeddings[triplets[:, A]]
        positives = embeddings[triplets[:, P]]
        negatives = embeddings[triplets[:, N]]
        ap_distances = torch.diagonal(torch.cdist(anchor, positives))
        an_distances = torch.diagonal(torch.cdist(anchor, negatives))
        pn_distances = torch.diagonal(torch.cdist(positives, negatives))
        #an_pn = torch.minimum(an_distances, pn_distances)
        #losses = F.relu(ap_distances - an_pn + self.margin)
        
        farther_apart_distances = an_distances + pn_distances
        losses = F.relu(ap_distances - torch.div(farther_apart_distances, 2) + self.margin) 

        return losses.mean(), len(triplets)


class OnlineTripletLossV5(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    Florian et al
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLossV5, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        A = 0
        P = 1
        N = 2
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
	
        anchor = embeddings[triplets[:, A]]
        positives = embeddings[triplets[:, P]]
        negatives = embeddings[triplets[:, N]]
        ap_distances = torch.diagonal(torch.cdist(anchor, positives))
        an_distances = torch.diagonal(torch.cdist(anchor, negatives)) 
        losses = F.relu(ap_distances - an_distances + self.margin)
 
        return losses.mean(), len(triplets)





