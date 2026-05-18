"""
Lightning callback that uploads the latest checkpoint to the Hugging Face Hub
every N training steps. Designed for RunPod spot instances — if the spot GPU is
preempted, training resumes from the most-recently-uploaded HF Hub checkpoint
rather than losing progress.

Configure via env vars before launching:
    HF_HUB_REPO_ID=<user>/<repo>          # required
    HF_TOKEN=<your_hf_token>              # required (write access)

The callback silently no-ops if repo_id is None — safe to leave in configs for
toy/CI runs.
"""
import os
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl


class HFHubUploadCallback(pl.Callback):
    def __init__(
        self,
        repo_id: Optional[str] = None,
        every_n_train_steps: int = 5000,
        path_in_repo: str = 'checkpoints',
        ckpt_filename: str = 'last.ckpt',
    ):
        super().__init__()
        self.repo_id = repo_id
        self.every_n = every_n_train_steps
        self.path_in_repo = path_in_repo
        self.ckpt_filename = ckpt_filename
        self._api = None  # lazy

    def _get_api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=os.environ.get('HF_TOKEN'))
            # Ensure the repo exists (create_repo is idempotent).
            self._api.create_repo(
                repo_id=self.repo_id, exist_ok=True, private=False)
        return self._api

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if self.repo_id is None:
            return
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if step == 0 or step % self.every_n != 0:
            return

        # Lightning's ModelCheckpoint writes last.ckpt; find it via configured callbacks.
        ckpt_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.last_model_path:
                ckpt_path = cb.last_model_path
                break
        if ckpt_path is None or not Path(ckpt_path).exists():
            return  # nothing to upload yet

        try:
            api = self._get_api()
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f'{self.path_in_repo}/step_{step}.ckpt',
                repo_id=self.repo_id,
                commit_message=f'auto: ckpt at step {step}',
            )
            # Also push as 'last.ckpt' for easy resume.
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f'{self.path_in_repo}/{self.ckpt_filename}',
                repo_id=self.repo_id,
                commit_message=f'auto: update last.ckpt to step {step}',
            )
        except Exception as e:
            # Never crash training because of an upload failure.
            print(f'[HFHubUploadCallback] upload at step {step} failed: {e}')
