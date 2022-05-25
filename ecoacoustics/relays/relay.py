from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

import attr
from conduit.data import CdtDataModule
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ranzen.decorators import implements
from ranzen.hydra import Option, Relay
import torch
from torch import nn

from ecoacoustics.manual_confs.logger_conf import WandbLoggerConf

__all__ = [
    "EcoacousticsRelay",
]


@attr.define(kw_only=True)
class EcoacousticsRelay(Relay):
    dm: DictConfig
    model: DictConfig
    trainer: DictConfig
    logger: DictConfig

    seed: Optional[int] = 42
    arftifact_dir: str = ".."

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        clear_cache: bool = False,
        dm: list[type[Any] | Option],
        model: list[type[Any] | Option],
    ) -> None:
        configs = dict(
            dm=dm,
            model=model,
            trainer=[Option(class_=pl.Trainer, name="base")],
            logger=[Option(class_=WandbLoggerConf, name="base")],
        )
        super().with_hydra(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=False,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: Dict[str, Any] | None = None) -> None:
        self.log(f"Current working directory: '{os.getcwd()}'")
        pl.seed_everything(self.seed, workers=True)

        dm: CdtDataModule = instantiate(self.dm)
        dm.prepare_data()
        dm.setup()

        # Rather than redownloading the model weights each time...
        weights_file = Path(__file__).parent.parent / "vggish_weights.pt"
        if not weights_file.exists():
            loaded: nn.Module = torch.hub.load(
                'DavidHurst/torchvggish',
                'vggish',
                preprocess=False,
                postprocess=False,
            )
            # model = DownloadWrapper(loaded, weights_file)
            torch.save({"state_dict": loaded.state_dict()}, weights_file)

        model: pl.LightningModule = instantiate(self.model)
        chkpt = torch.load(weights_file)
        model.load_state_dict(chkpt["state_dict"])

        # Create a Logger
        if self.logger.get("group", None) is None:
            default_group = f"{dm.__class__.__name__}"
            self.logger["group"] = default_group
        logger: WandbLogger = instantiate(self.logger, reinit=True)
        if raw_config is not None:
            logger.log_hyperparams(raw_config)  # type: ignore

        trainer: pl.Trainer = instantiate(self.trainer, logger=logger)
        # Fine tune the vggish.embedding layers
        trainer.fit(model, dm)

        trainer.predict(model, dm)
