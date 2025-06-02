import torch


def make_optimizer(model, config):
    try:
        warm_optimizer = torch.optim.Adam(
            [
                {
                    "params": model._backbone_params,
                    "lr": config["warm_optimizer_lr"],
                    "initial_lr": config["warm_optimizer_lr"],
                    "weight_decay": 0,
                    "fused": True,
                },
                {
                    "params": model._last_backbone_block_params,
                    "lr": config["warm_optimizer_lr"],
                    "initial_lr": config["warm_optimizer_lr"],
                    "weight_decay": 0,
                    "fused": True,
                },
            ]
        )
    except KeyError:
        warm_optimizer = None

    try:
        joint_optimizer = torch.optim.Adam(
            [
                {
                    "params": model._backbone_params,
                    "lr": config["joint_optimizer_lr.backbone"],
                    "initial_lr": config["joint_optimizer_lr.backbone"],
                    "weight_decay": 0,
                    "fused": True,
                },
                {
                    "params": model._last_backbone_block_params,
                    "lr": config["joint_optimizer_lr.prototype_layers"],
                    "initial_lr": config["joint_optimizer_lr.prototype_layers"],
                    "weight_decay": 0,
                    "fused": True,
                },
                {
                    "params": model.last_layer.parameters(),
                    "lr": config["joint_optimizer_lr.last_layer"],
                    "initial_lr": config["joint_optimizer_lr.last_layer"],
                    "weight_decay": 0,
                    "fused": True,
                },
            ]
        )
    except KeyError:
        joint_optimizer = None

    try:
        fine_tune_optimizer = torch.optim.Adam(
            [
                {
                    "params": model.last_layer.parameters(),
                    "lr": config["fine_tune_optimizer_lr"],
                    "initial_lr": config["fine_tune_optimizer_lr"],
                    "weight_decay": 0,
                    "fused": True,
                },
            ]
        )
    except KeyError:
        fine_tune_optimizer = None

    return warm_optimizer, joint_optimizer, fine_tune_optimizer
