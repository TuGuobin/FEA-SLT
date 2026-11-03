"""Train, resume, validate or test FEA-SLT with explicit configuration."""

import argparse
import datetime
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from feaslt.callbacks import SetupCallback
from utils.helpers import instantiate_from_config


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", nargs="+", default=[])
    for name, default in (("train", None), ("val", False), ("test", False), ("no_test", True)):
        parser.add_argument(f"--{name}", type=str2bool, nargs="?", const=True, default=default)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-f", "--fast_dev_run", action="store_true")
    parser.add_argument("-n", "--name", default="")
    parser.add_argument("--postfix", default="")
    parser.add_argument("-l", "--logdir", default="logs")
    parser.add_argument("-r", "--resume", help="Existing run directory; restores optimizer and scheduler.")
    parser.add_argument("--ckpt", help="Checkpoint path; with --resume, may be a name under checkpoints/.")
    parser.add_argument("-e", "--evaluation", choices=("bleu", "loss"),
                        help="Checkpoint criterion; defaults to the saved run setting.")
    parser.add_argument("--logger", choices=("csv", "tensorboard", "wandb", "none"), default="csv")
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--limit_test_batches", type=int)
    return parser


def setup_logging_dirs(opt):
    if opt.resume:
        logdir = Path(opt.resume)
        if not logdir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {logdir}")
        ckpt = Path(opt.ckpt or "last.ckpt")
        if not ckpt.is_absolute():
            ckpt = logdir / "checkpoints" / ckpt
    else:
        suffix = opt.name or (Path(opt.config[0]).stem if opt.config else "evaluation")
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        logdir = Path(opt.logdir) / f"{now}_{suffix}{opt.postfix}"
        ckpt = Path(opt.ckpt) if opt.ckpt else None
    if ckpt is not None and not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return str(logdir), str(ckpt) if ckpt else None, logdir.name


def configure_logger(logger_type, logdir, name):
    if logger_type == "none":
        return False
    targets = {"csv": "CSVLogger", "tensorboard": "TensorBoardLogger", "wandb": "WandbLogger"}
    params = {"save_dir": logdir, "name": "metrics"}
    if logger_type == "wandb":
        params.update(name=name, project="FEA-SLT")
    return instantiate_from_config({
        "target": f"pytorch_lightning.loggers.{targets[logger_type]}", "params": params,
    })


def main():
    parser = get_parser()
    opt = parser.parse_args()
    opt.train = not (opt.val or opt.test) if opt.train is None else opt.train
    if opt.train and (opt.val or opt.test):
        parser.error("Choose training or validation/testing.")
    if not (opt.train or opt.val or opt.test):
        parser.error("Choose --train, --val or --test.")
    if opt.name and opt.resume:
        parser.error("--name and --resume cannot be combined.")
    logdir, ckpt, nowname = setup_logging_dirs(opt)
    if not opt.train and ckpt is None:
        parser.error("Validation/testing requires --ckpt or --resume.")

    paths = []
    if opt.resume:
        paths = sorted((Path(opt.resume) / "configs").glob("*.yaml"))
    elif not opt.config and ckpt:
        paths = sorted((Path(ckpt).parent.parent / "configs").glob("*.yaml"))
    paths += [Path(path) for path in opt.config]
    if not paths:
        parser.error("Provide --config (or a run directory with saved configs).")
    config = OmegaConf.merge(*[OmegaConf.load(path) for path in paths])
    lightning = config.pop("lightning", OmegaConf.create())
    trainer_config = dict(OmegaConf.to_container(lightning.get("trainer", {}), resolve=True))
    devices = trainer_config.get("devices", 1)
    if devices != 1 and not (isinstance(devices, list) and len(devices) == 1):
        raise ValueError("This release supports a single device; distributed metrics are not implemented.")
    if trainer_config.get("num_nodes", 1) != 1:
        raise ValueError("This release supports a single node.")
    trainer_config["fast_dev_run"] = opt.fast_dev_run
    for key in ("limit_val_batches", "limit_test_batches"):
        value = getattr(opt, key)
        if value is not None:
            if value < 1:
                parser.error(f"--{key} must be positive.")
            trainer_config[key] = value
    lightning.trainer = trainer_config
    seed_everything(opt.seed, workers=True)
    data = instantiate_from_config(config.data)
    model = instantiate_from_config(config.model)
    model.logdir = logdir
    callbacks = []
    if opt.train:
        if opt.evaluation == "bleu":
            model.monitor = "val/bleu4"
        elif opt.evaluation == "loss":
            model.monitor = "val/contra_loss" if model.cross_modal_align and not model.combined_loss and model.warm_up_steps is None else "val/loss"
        else:
            model.monitor = lightning.get("checkpoint", {}).get("monitor", model.monitor)
        mode = "max" if model.monitor.startswith("val/bleu") or model.monitor.startswith("val/rouge") else "min"
        lightning.checkpoint = {"monitor": model.monitor, "mode": mode}
        config.model.params.monitor = model.monitor
        for callback in lightning.get("callback", {}).values():
            if opt.logger == "none" and callback.target.endswith("LearningRateMonitor"):
                continue
            callbacks.append(instantiate_from_config(callback))
        callbacks.extend([
            ModelCheckpoint(
                dirpath=str(Path(logdir) / "checkpoints"), filename="epoch={epoch:04d}-step={step:07d}",
                auto_insert_metric_name=False, monitor=model.monitor,
                mode=mode, save_top_k=1, save_last=True,
            ),
            SetupCallback(opt.resume, nowname, logdir, str(Path(logdir) / "checkpoints"),
                          str(Path(logdir) / "configs"), config, lightning),
        ])
    trainer_config.update(callbacks=callbacks, logger=configure_logger(opt.logger, logdir, nowname),
                          enable_checkpointing=opt.train)
    trainer = Trainer(**trainer_config)
    if opt.train:
        if ckpt and not opt.resume:
            model.load_pretrained_weights(ckpt)
        trainer.fit(model, datamodule=data, ckpt_path=ckpt if opt.resume else None)
        if not opt.no_test:
            trainer.test(model, datamodule=data, ckpt_path="best")
    else:
        if opt.val:
            trainer.validate(model, datamodule=data, ckpt_path=ckpt)
        if opt.test:
            trainer.test(model, datamodule=data, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
