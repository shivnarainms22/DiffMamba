"""
Project-local Lightning callbacks. Keep this module small; the heavy HF Hub
logic lives in `hf_hub_callback.py`.
"""
import os
import signal
from pathlib import Path

import lightning.pytorch as pl


class SigtermCheckpointCallback(pl.Callback):
    """
    On SIGTERM (RunPod spot-eviction signal, ~30s warning), write a final
    checkpoint before the pod dies. Lightning's default SignalConnector only
    handles SLURM signals; spot platforms send SIGTERM directly.

    This callback installs a handler at fit-start that flips a flag; the next
    batch end triggers the save. We can't save synchronously inside the signal
    handler (CUDA/Lightning state is not signal-safe), so we cooperate with
    the training loop and bail out at the next batch boundary.
    """

    def __init__(self, save_filename: str = 'sigterm_save.ckpt'):
        super().__init__()
        self.save_filename = save_filename
        self._sigterm_received = False
        self._prev_handler = None

    def on_fit_start(self, trainer, pl_module):
        self._prev_handler = signal.signal(signal.SIGTERM, self._handle)

    def on_fit_end(self, trainer, pl_module):
        if self._prev_handler is not None:
            try:
                signal.signal(signal.SIGTERM, self._prev_handler)
            except Exception:
                pass

    def _handle(self, signum, frame):
        # Keep this minimal — signal-safe operations only.
        self._sigterm_received = True
        print(f'\n[SigtermCheckpointCallback] SIGTERM received, will save at next batch boundary.\n',
              flush=True)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if not self._sigterm_received:
            return
        if trainer.global_rank != 0:
            trainer.should_stop = True
            return
        # Save once and request graceful stop.
        try:
            save_dir = Path(trainer.default_root_dir) / 'checkpoints'
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / self.save_filename
            trainer.save_checkpoint(str(path))
            print(f'[SigtermCheckpointCallback] saved {path} at step {trainer.global_step}',
                  flush=True)
        except Exception as e:
            print(f'[SigtermCheckpointCallback] save failed: {e}', flush=True)
        finally:
            trainer.should_stop = True
