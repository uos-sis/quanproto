import torch
from tqdm import tqdm

from quanproto.metrics import general


def evaluate_threshold(model, dataloader):
    threshold = 0
    total_logits = torch.empty(0).cuda()
    total_labels = torch.empty(0).cuda()

    results = {}

    test_iter = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Threshold Evaluation",
        mininterval=2.0,
        ncols=0,
    )

    model.eval()
    for _, (inputs, labels) in test_iter:
        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            logits = model.predict(inputs)

        total_logits = torch.cat((total_logits, logits), dim=0)
        total_labels = torch.cat((total_labels, labels), dim=0)
        del logits, inputs, labels

    mean_tp_activation = general.mean_tp_activation(total_logits, total_labels)
    mean_tn_activation = general.mean_tn_activation(total_logits, total_labels)
    threshold = (mean_tp_activation + mean_tn_activation) / 2

    print(f"Threshold: {threshold}")
    results["Threshold"] = threshold

    return results
