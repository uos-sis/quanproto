import torch
import sklearn.metrics as metrics
from quanproto.metrics.helpers import label_prediction


def roc_curve(targets, logits):
    fpr = dict()
    tpr = dict()

    multi_label = check_multi_label(targets)
    if not multi_label:
        num_classes = targets.max().item() + 1
    else:
        num_classes = targets.shape[1]

    for i in range(num_classes):
        if multi_label:
            # the multi label case has targets with shape (n_samples, n_classes)
            fpr[i], tpr[i], _ = metrics.roc_curve(targets[:, i], logits[:, i])
        else:
            # the single label case has targets with shape (n_samples,)
            fpr[i], tpr[i], _ = metrics.roc_curve(targets == i, logits[:, i])

    return {"fpr": fpr, "tpr": tpr}


def check_multi_label(targets: torch.Tensor):
    if targets.dim() == 1:
        multi_label = False
    else:
        multi_label = True
    return multi_label


def get_sample_weights(targets: torch.Tensor):
    num_samples = targets.shape[0]

    multi_label = check_multi_label(targets)

    if multi_label:
        # the multi label case has targets with shape (n_samples, n_classes)
        counts = targets.sum(axis=0)
        class_freq = counts / num_samples
        # we add a small value to avoid division by zero
        class_weights = 1 / (class_freq + 1e-8)
        sample_weights = torch.sum(targets * class_weights, axis=1)
    else:
        # the single label case has targets with shape (n_samples,)
        counts = torch.bincount(targets.int())
        class_freq = counts / num_samples
        # we add a small value to avoid division by zero
        class_weights = 1 / (class_freq + 1e-8)
        sample_weights = class_weights[targets.int()]

    return sample_weights


def mean_tp_activation(logits: torch.Tensor, targets: torch.Tensor, binary=False, balanced=False):
    multi_label = check_multi_label(targets)

    if multi_label:
        positive_logits = torch.where(targets.bool(), logits, torch.tensor(0.0))
        positive_logits = torch.sum(positive_logits, dim=1)

        positive_count = torch.sum(targets, dim=1)

        # check if positive_count is zero
        if torch.any(positive_count == 0):
            positive_count = positive_count + 1

        mean_positive_logits = torch.mean(positive_logits / positive_count)

        return mean_positive_logits.item()

    else:
        positive_logits = logits[targets]

        positive_count = 1

        mean_positive_logits = torch.mean(positive_logits / positive_count)

        return mean_positive_logits.item()


def mean_tn_activation(logits: torch.Tensor, targets: torch.Tensor, binary=False, balanced=False):
    multi_label = check_multi_label(targets)

    if multi_label:
        negative_logits = torch.where(~targets.bool(), logits, torch.tensor(0.0))
        negative_logits = torch.sum(negative_logits, dim=1)

        positive_count = torch.sum(targets, dim=1)
        negative_count = targets.shape[1] - positive_count

        mean_negative_logits = torch.mean(negative_logits / negative_count)

        return mean_negative_logits.item()

    else:
        negative_logits = logits[~targets]

        negative_count = logits.shape[1] - 1

        mean_negative_logits = torch.mean(negative_logits / negative_count)

        return mean_negative_logits.item()


def accuracy(
    logits: torch.Tensor, targets: torch.Tensor, binary=False, balanced=False, threshold=0.5
):
    if not binary:
        multi_label = check_multi_label(targets)
        predictions = label_prediction(logits, multi_label, threshold=threshold)
    else:
        predictions = logits

    acc = metrics.accuracy_score(
        targets.cpu().detach().numpy(),
        predictions.cpu().detach().numpy(),
        sample_weight=(
            None if not balanced else get_sample_weights(targets).cpu().detach().numpy()
        ),
    )
    return acc


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k=3,
    binary=False,
    balanced=False,
):
    multi_label = check_multi_label(targets)

    if multi_label:
        raise ValueError("Top-k accuracy is not supported for multi-label classification.")

    if binary:
        raise ValueError("Top-k accuracy is not supported for binary classification.")

    topk_acc = metrics.top_k_accuracy_score(
        targets.cpu().detach().numpy(),
        logits.cpu().detach().numpy(),
        k=k,
        sample_weight=(
            None if not balanced else get_sample_weights(targets).cpu().detach().numpy()
        ),
    )
    return topk_acc


def precision(
    logits: torch.Tensor, targets: torch.Tensor, binary=False, balanced=False, threshold=0.5
):
    multi_label = check_multi_label(targets)

    if not binary:
        predictions = label_prediction(logits, multi_label, threshold=threshold)
    else:
        predictions = logits

    pre = metrics.precision_score(
        targets.cpu().detach().numpy(),
        predictions.cpu().detach().numpy(),
        average="samples" if multi_label else "macro",
        zero_division=0,
        sample_weight=(
            None if not balanced else get_sample_weights(targets).cpu().detach().numpy()
        ),
    )
    return pre


def recall(
    logits: torch.Tensor, targets: torch.Tensor, binary=False, balanced=False, threshold=0.5
):
    multi_label = check_multi_label(targets)

    if not binary:
        predictions = label_prediction(logits, multi_label, threshold=threshold)
    else:
        predictions = logits

    re = metrics.recall_score(
        targets.cpu().detach().numpy(),
        predictions.cpu().detach().numpy(),
        average="samples" if multi_label else "macro",
        zero_division=0,
        sample_weight=(
            None if not balanced else get_sample_weights(targets).cpu().detach().numpy()
        ),
    )
    return re


def f1_score(
    logits: torch.Tensor, targets: torch.Tensor, binary=False, balanced=False, threshold=0.5
):
    multi_label = check_multi_label(targets)

    if not binary:
        predictions = label_prediction(logits, multi_label, threshold=threshold)
    else:
        predictions = logits

    f1 = metrics.f1_score(
        targets.cpu().detach().numpy(),
        predictions.cpu().detach().numpy(),
        average="samples" if multi_label else "macro",
        zero_division=0,
        sample_weight=(
            None if not balanced else get_sample_weights(targets).cpu().detach().numpy()
        ),
    )
    return f1


def roc_auc_score(logits: torch.Tensor, targets: torch.Tensor, balanced=False):
    multi_label = check_multi_label(targets)
    if not multi_label:
        logits = torch.softmax(logits, dim=1)

    roc = metrics.roc_auc_score(
        targets.cpu().detach().numpy(),
        logits.cpu().detach().numpy(),
        average="samples" if multi_label else "macro",
        multi_class="ovr",
        sample_weight=(
            None if not balanced else get_sample_weights(targets).cpu().detach().numpy()
        ),
    )
    return roc
