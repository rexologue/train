from __future__ import annotations


def sequence_logprobs(logits, labels, ignore_index: int):
    import torch

    logp = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    gathered = logp.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)

