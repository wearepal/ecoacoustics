from types import MethodType
from typing import TypeVar

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from conduit.data import BinarySample
from ranzen import parsable, implements

T = TypeVar('T')


class Vggish(pl.LightningModule):
    @parsable
    def __init__(self):
        super(Vggish, self).__init__()
        self.vggish = torch.hub.load(
            "DavidHurst/torchvggish",
            "vggish",
            pretrained=True,
            preprocess=False,
            postprocess=False,
        )
        # VGGish comprises two main blocks, [features, embeddings]
        for param in self.vggish.features.parameters():
            param.requires_grad = False
        self.vggish.features.eval()

    def step(self, batch: BinarySample, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.vggish(x.float())  # VGGish only takes Float, not Double
        return F.cross_entropy(y_hat, y)

    @implements(pl.LightningModule)
    def training_step(self, batch: BinarySample, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx)

    @implements(pl.LightningModule)
    def validation_step(self, batch: BinarySample, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Only train the embeddings network
        return torch.optim.Adam(self.vggish.embeddings.parameters(), lr=0.02)
