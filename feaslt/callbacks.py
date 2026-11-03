import os

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback


class SetupCallback(Callback):
    """Save reproducible configuration snapshots and an emergency checkpoint."""

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0 and pl_module.global_step != 0:
            os.makedirs(self.ckptdir, exist_ok=True)
            trainer.save_checkpoint(os.path.join(self.ckptdir, "last.ckpt"))

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            for directory in (self.logdir, self.ckptdir, self.cfgdir):
                os.makedirs(directory, exist_ok=True)
            # Resolve path/environment interpolations so resuming uses the original inputs.
            OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"), resolve=True)
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"), resolve=True,
            )
