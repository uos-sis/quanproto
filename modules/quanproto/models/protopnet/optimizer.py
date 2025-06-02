import torch


def make_optimizer(model, config):
    try:
        warm_optimizer = torch.optim.Adam(
            [
                {
                    "params": model.add_on_layers.parameters(),
                    "lr": config["warm_optimizer_lr"],
                    "initial_lr": config["warm_optimizer_lr"],
                    "weight_decay": 0,
                    "fused": True,
                },
                {
                    "params": model.prototype_vectors,
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
                    "params": model.backbone.parameters(),
                    "lr": config["joint_optimizer_lr.backbone"],
                    "initial_lr": config["joint_optimizer_lr.backbone"],
                    "weight_decay": 0,
                    "fused": True,
                },
                {
                    "params": model.add_on_layers.parameters(),
                    "lr": config["joint_optimizer_lr.add_on_layers"],
                    "initial_lr": config["joint_optimizer_lr.add_on_layers"],
                    "weight_decay": 0,
                    "fused": True,
                },
                {
                    "params": model.prototype_vectors,
                    "lr": config["joint_optimizer_lr.prototype_vectors"],
                    "initial_lr": config["joint_optimizer_lr.prototype_vectors"],
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
