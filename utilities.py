import logging
import torch


def _optimizer(
    params,
    type: str = "sgd",
    lr: float = 0.2,
    momentum: float = 0.9,
    weight_decay: float = 0.0000015,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    **kwargs,
):

    betas = (beta_1, beta_2)

    # for adam, lr is the step size and is modified by exp. moving av. of prev. gradients
    opts = {
        "adam": lambda p: torch.optim.Adam(p, lr=lr, betas=betas, weight_decay=weight_decay),
        "adamw": lambda p: torch.optim.AdamW(p, lr=lr, betas=betas, weight_decay=weight_decay),
        "sgd": lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum, weight_decay=weight_decay),
    }

    if type == "adam" and lr > 0.01:
        logging.warning(f"Learning rate {lr} may be too high for adam")

    opt = opts[type](params)

    return opt

    # Apply LARS wrapper if option is chosen
    # if config["optimizer"]["lars"]:
    #     opt = LARSWrapper(opt, eta=config["optimizer"]["trust_coef"])
