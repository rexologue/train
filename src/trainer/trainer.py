from __future__ import annotations


class RoutedTrainer:
    def compute_loss(self, model, batch):
        loss_kind = batch.get("loss_kind")
        if loss_kind in {"sft_target", "sft_tool"}:
            from losses.sft import sft_cross_entropy_loss

            return sft_cross_entropy_loss(model, batch)
        if loss_kind == "dpo_target":
            raise NotImplementedError("DPO route requires chosen/rejected logprob plumbing")
        raise ValueError(f"unknown loss_kind {loss_kind!r}")

