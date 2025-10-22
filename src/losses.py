import torch
import json

import lightning as L
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from __future__ import division, print_function
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

def find_intersection(mu1, sigma1, mu2, sigma2):
    # Handle special case where variances are equal
    if sigma1 == sigma2:
        raise ValueError("The variances must be different for distinct intersection points.")
    
    a = sigma2**2 - sigma1**2
    b = mu1 * sigma2**2 - mu2 * sigma1**2
    c = (mu1 - mu2)**2 + 2 * (sigma1**2 - sigma2**2) * np.log(sigma2 / sigma1)
    
    discriminant = sigma1 * sigma2 * np.sqrt(c)
    
    # Compute the two intersection points
    x1 = (b + discriminant) / a
    x2 = (b - discriminant) / a
    
    if x1 >= min(mu1, mu2) and x1 <= max(mu1, mu2):
        return x1
    return x2

def get_probability_of_correct_class(logits, labels,):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return log_p_t.exp()

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes="present", ignore_index=-100):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    # mask = labels != ignore_index
    # probas = probas * mask.unsqueeze(1)
    # labels = labels * mask

    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)  # Foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        # if C == 1:
        #     if len(classes) > 1:
        #         raise ValueError("Sigmoid output possible only with 1 class")
        #     class_pred = probas[:, 0]
        # else:
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)

def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        # Assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def focal_loss(logits, labels, alpha=1.0, gamma=2):
    # Move labels to the correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the correct labels
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()

    # Compute the modulating factor
    modulating_factor = alpha * (1 - p_t) ** gamma
    
    # Stop gradient for modulating factor
    # modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.
    
    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t
    # Take the mean loss over all tokens and batches
    loss = loss.mean()
    return loss

def focal_clipped_loss(logits, labels, alpha=1.0, gamma=2):
    # Move labels to the correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the correct labels
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()

    # Compute the modulating factor
    modulating_factor = (1 - p_t) ** gamma
    # Compute the modulating factor
    modulating_factor_clipped = torch.where(modulating_factor > 0.99, torch.tensor(0.0, dtype=modulating_factor.dtype, device=modulating_factor.device), torch.tensor(1.0, dtype=modulating_factor.dtype, device=modulating_factor.device))
    modulating_factor = alpha * modulating_factor_clipped
   
    # Stop gradient for modulating factor
    modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.
    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t
    # Take the mean loss over all tokens and batches
    loss = loss.mean()
    return loss

def entropy_loss(logits, labels, alpha=1.0, gamma=2):
    # Move labels to the correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the correct labels
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()
    # Compute the modulating factor
    modulating_factor = alpha * (1 - gamma * p_t)
    # Stop gradient for modulating factor
    modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.
    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t    
    # Take the mean loss over all tokens and batches
    loss = loss.mean()
    return loss

def clipped_loss(logits, labels, alpha=1.0, gamma=0.8):
    # Move labels to the correct device to enable model parallelism
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    # Logits shape: (Batch, T, Vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather the log probabilities of the correct labels
    log_p_t = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) 
    # Compute probabilities of the correct labels
    p_t = log_p_t.exp()
    # Compute the modulating factor
    p_t_clipped = torch.where(p_t > gamma, torch.tensor(0.0, dtype=p_t.dtype, device=p_t.device), torch.tensor(1.0, dtype=p_t.dtype, device=p_t.device))
    modulating_factor = alpha * p_t_clipped
    
    # Stop gradient for modulating factor
    modulating_factor = modulating_factor.detach() # Look into this to check if it is correct.

    # Compute the focal loss per token
    loss = -modulating_factor * log_p_t    
    # Take the mean loss over all tokens and batches
    loss = loss.mean()
    return loss