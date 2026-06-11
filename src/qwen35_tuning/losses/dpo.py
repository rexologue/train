from __future__ import annotations


def dpo_loss_from_logratios(chosen_logratio, rejected_logratio, beta: float):
    import torch

    return -torch.nn.functional.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()

