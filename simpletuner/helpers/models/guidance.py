"""Classifier-free velocity guidance shared by image flow pipelines."""


def guided_velocity(positive, negative, scale, zero_star=False):
    positive, negative = positive.float(), negative.float()
    if zero_star:
        p, n = positive.flatten(1), negative.flatten(1)
        alpha = (p * n).sum(1) / n.square().sum(1).clamp_min(1e-8)
        negative = negative * alpha.view(-1, *([1] * (negative.ndim - 1)))
    return negative + scale * (positive - negative)
