from __future__ import annotations
from typing import Any, List

from conduit.data import EcoacousticsDataModule
from ranzen.hydra import Option
from typing_extensions import Type

from ecoacoustics.models.vggish import Vggish
from ecoacoustics.relay import EcoacousticsRelay

if __name__ == "__main__":
    dm_ops: List[Type[Any] | Option] = [
        Option(EcoacousticsDataModule, name="ecoacoustics"),
    ]
    model_ops: List[Type[Any] | Option] = [
        Option(Vggish, name="vggish"),
    ]

    EcoacousticsRelay.with_hydra(
        root="conf",
        dm=dm_ops,
        model=model_ops,
        clear_cache=True,
    )
