# eta_callback.py
import time, datetime
import pytorch_lightning as pl

class ETACallback(pl.Callback):
    def __init__(self, print_every=50):
        self.print_every = print_every
        self.t0 = None
        self.total_steps = None

    def on_fit_start(self, trainer, pl_module):
        self.t0 = time.perf_counter()
        # Lightning computes this for you (respects max_steps/epochs, dataset, etc.)
        self.total_steps = trainer.estimated_stepping_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = trainer.global_step
        if s == 0:
            return
        elapsed = time.perf_counter() - self.t0
        sps = s / max(elapsed, 1e-6)
        rem = max(self.total_steps - s, 0)
        eta_sec = rem / max(sps, 1e-9)
        # Put key numbers into the progress bar
        pl_module.log_dict(
            {"steps_per_sec": sps, "eta_min": eta_sec / 60.0},
            prog_bar=True, on_step=True, logger=True
        )
        # Occasionally print a readable ETA line
        if s % self.print_every == 0:
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            trainer.strategy.barrier()  # be polite in DDP
            if trainer.is_global_zero:
                print(f"[ETA] step {s}/{self.total_steps} | {sps:.2f} steps/s | ~{eta_str} remaining")
