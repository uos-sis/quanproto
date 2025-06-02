import torch


def get_scheduler(warm_optimizer, joint_optimizer, fine_tune_optimizer, config, mode="min"):
    warm_lr_scheduler = None
    joint_lr_scheduler = None
    fine_tune_lr_scheduler = None

    find_warmup = False
    find_joint = False
    find_fine_tune = False

    for key in config.keys():
        if key.startswith("warm_scheduler"):
            find_warmup = True
        if key.startswith("joint_scheduler"):
            find_joint = True
        if key.startswith("fine_tune_scheduler"):
            find_fine_tune = True

    if find_warmup and warm_optimizer is not None:
        not_found = False
        try:
            warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                warm_optimizer,
                step_size=config["warm_scheduler.step_size"],
                gamma=config["warm_scheduler.gamma"],
            )
        except KeyError:
            not_found = True
        if not_found:
            try:
                warm_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    warm_optimizer,
                    milestones=config["warm_scheduler.milestones"],
                    gamma=config["warm_scheduler.gamma"],
                )
                not_found = False
            except KeyError:
                not_found = True
        if not_found:
            try:
                warm_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    warm_optimizer,
                    mode=mode,  # min for loss, max for accuracy
                    factor=config["warm_scheduler.factor"],
                    patience=config["warm_scheduler.patience"],
                )
                not_found = False
            except KeyError:
                not_found = True
        if not_found:
            raise ValueError("No valid warm_scheduler found")

    if find_joint and joint_optimizer is not None:
        not_found = False
        try:
            joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=config["joint_scheduler.step_size"],
                gamma=config["joint_scheduler.gamma"],
            )
        except KeyError:
            not_found = True

        if not_found:
            try:
                joint_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    joint_optimizer,
                    milestones=config["joint_scheduler.milestones"],
                    gamma=config["joint_scheduler.gamma"],
                    last_epoch=config["warmup_epochs"],
                )
                not_found = False
            except KeyError as e:
                print(e)
                not_found = True

        if not_found:
            try:
                joint_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    joint_optimizer,
                    mode=mode,  # min for loss, max for accuracy
                    factor=config["joint_scheduler.factor"],
                    patience=config["joint_scheduler.patience"],
                )
                not_found = False
            except KeyError:
                not_found = True
        if not_found:
            raise ValueError("No valid joint_scheduler found")

    if find_fine_tune and fine_tune_optimizer is not None:
        not_found = False
        try:
            fine_tune_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                fine_tune_optimizer,
                step_size=config["fine_tune_scheduler.step_size"],
                gamma=config["fine_tune_scheduler.gamma"],
            )
        except KeyError:
            not_found = True

        if not_found:
            try:
                fine_tune_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    fine_tune_optimizer,
                    milestones=config["fine_tune_scheduler.milestones"],
                    gamma=config["fine_tune_scheduler.gamma"],
                    last_epoch=config["warmup_epochs"] + config["joint_epochs"],
                )
                not_found = False
            except KeyError:
                not_found = True

        if not_found:
            try:
                fine_tune_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    fine_tune_optimizer,
                    mode=mode,  # min for loss, max for accuracy
                    factor=config["fine_tune_scheduler.factor"],
                    patience=config["fine_tune_scheduler.patience"],
                )
                not_found = False
            except KeyError:
                not_found = True
        if not_found:
            raise ValueError("No valid fine_tune_scheduler found")

    return warm_lr_scheduler, joint_lr_scheduler, fine_tune_lr_scheduler
