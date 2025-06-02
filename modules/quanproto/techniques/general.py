import torch
from tqdm import tqdm

import quanproto.metrics.general


def evaluate_general(
    model,
    dataloader,
    balanced=True,
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
):

    total_logits = torch.empty(0).cuda()
    total_labels = torch.empty(0).cuda()

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="General Evaluation",
        mininterval=2.0,
        ncols=0,
    )

    model.eval()
    for i, (batch) in test_iter:
        if len(batch) == 2:
            inputs, target = batch
        if len(batch) == 3:
            inputs = torch.cat([batch[0], batch[1]])
            target = torch.cat([batch[2], batch[2]])

        inputs = inputs.cuda()
        target = target.cuda()

        with torch.no_grad():
            logits = model.predict(inputs)

        total_logits = torch.cat((total_logits, logits), dim=0)
        total_labels = torch.cat((total_labels, target), dim=0)
        del logits, inputs, target

    binary = False
    results = {}

    if "accuracy" in metrics:
        results["Accuracy"] = (
            quanproto.metrics.general.accuracy(
                total_logits,
                total_labels,
                binary=binary,
                balanced=balanced,
                threshold=model.multi_label_threshold,
            )
            * 100
        )

    if model.multi_label is False and "top-3 accuracy" in metrics:
        results["top-3 Accuracy"] = (
            quanproto.metrics.general.topk_accuracy(
                total_logits, total_labels, k=3, balanced=balanced
            )
            * 100
        )

    if "precision" in metrics:
        results["Precision"] = (
            quanproto.metrics.general.precision(
                total_logits,
                total_labels,
                binary=binary,
                balanced=balanced,
                threshold=model.multi_label_threshold,
            )
            * 100
        )

    if "recall" in metrics:
        results["Recall"] = (
            quanproto.metrics.general.recall(
                total_logits,
                total_labels,
                binary=binary,
                balanced=balanced,
                threshold=model.multi_label_threshold,
            )
            * 100
        )

    if "f1" in metrics:
        results["F1 score"] = (
            quanproto.metrics.general.f1_score(
                total_logits,
                total_labels,
                binary=binary,
                balanced=balanced,
                threshold=model.multi_label_threshold,
            )
            * 100
        )

    if "roc_auc" in metrics:
        results["ROC AUC"] = (
            quanproto.metrics.general.roc_auc_score(
                total_logits, total_labels, balanced=balanced
            )
            * 100
        )

    return results
