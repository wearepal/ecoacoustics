from types import MethodType
from typing import TypeVar

import pytorch_lightning as pl
import torch
from conduit.data import BinarySample
from ranzen import parsable, implements
from torch import nn

T = TypeVar('T')


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Vggish(pl.LightningModule):
    @parsable
    def __init__(self):
        super().__init__()

        self.features = make_layers()
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )

        # VGGish comprises two main blocks, [features, embeddings]
        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def step(self, batch: BinarySample, batch_idx: int) -> torch.Tensor:
        x, y = batch
        self.features.eval()
        with torch.no_grad():
            x = self.features(x.float())

            # Transpose the output from features to
            # remain compatible with vggish embeddings
            x = torch.transpose(x, 1, 3)
            x = torch.transpose(x, 1, 2)
            x = x.contiguous()
            x = x.view(x.size(0), -1)

        y_hat = self.embeddings(x)
        return self._loss(y_hat, y.long())

    @implements(pl.LightningModule)
    def training_step(self, batch: BinarySample, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx)

    @implements(pl.LightningModule)
    def validation_step(self, batch: BinarySample, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Only train the embeddings network
        return torch.optim.Adam(self.embeddings.parameters(), lr=0.02)
