from __future__ import annotations


def validation_marker(name: str) -> str:
    return f"validation:{name}:start"


class TrainingProgress:
    def __init__(self, *, total_steps: int, enabled: bool = True, main_process: bool = True):
        self.enabled = enabled and main_process
        self.total_steps = total_steps
        self._bar = None
        self._last_step = 0

    def __enter__(self) -> "TrainingProgress":
        if self.enabled:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=self.total_steps, desc="train", dynamic_ncols=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._bar is not None:
            self._bar.close()

    def metrics(self, metrics: dict, state) -> None:
        if not self.enabled or self._bar is None:
            return
        if state.global_step > self._last_step:
            self._bar.update(state.global_step - self._last_step)
            self._last_step = state.global_step
        postfix = {}
        for key in (
            "train/loss",
            "train/lr",
            "train/samples_per_second",
            "train/tokens_per_second",
            "train/supervised_tokens_per_second",
            "eval/loss",
            "eval/bfcl/accuracy",
        ):
            if key in metrics:
                postfix[key.rsplit("/", 1)[-1]] = f"{float(metrics[key]):.4g}"
        if postfix:
            self._bar.set_postfix(postfix)

    def phase(self, name: str, state) -> None:
        if not self.enabled:
            return
        messages = {
            "validation:standard:start": f"[validation {state.validation_index:04d}] standard eval started",
            "validation:bfcl:start": f"[validation {state.validation_index:04d}] bfcl eval started",
            "checkpoint:save:start": f"[checkpoint {state.checkpoint_index:04d}] save started",
            "validation:end": f"[validation {state.validation_index:04d}] finished",
        }
        message = messages.get(name)
        if message:
            if self._bar is not None:
                self._bar.write(message)
            else:
                print(message)
