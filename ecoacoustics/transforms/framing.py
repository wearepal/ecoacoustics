import numpy as np
import torch
from torch import nn

__all__ = ["FramingNp"]


class FramingNp(nn.Module):
    def __init__(
        self,
        stft_hop_length_seconds: float = 0.010,
        example_window_seconds: float = 0.96,  # Each example contains 96 10ms frames
        example_hop_seconds: float = 0.96,  # with zero overlap.
    ):
        super().__init__()
        self.stft_hop_length_seconds = stft_hop_length_seconds
        self.example_window_seconds = example_window_seconds
        self.example_hop_seconds = example_hop_seconds

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Frame features into examples
        np_data = data.numpy()
        features_sample_rate = 1.0 / self.stft_hop_length_seconds
        example_window_length = int(round(self.example_window_seconds * features_sample_rate))
        example_hop_length = int(round(self.example_hop_seconds * features_sample_rate))
        log_mel_examples = self.frame(
            np_data, window_length=example_window_length, hop_length=example_hop_length
        )

        return torch.as_tensor(log_mel_examples, device=data.device).unsqueeze(1)

    def frame(self, data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
        """Convert array into a sequence of successive possibly overlapping frames.
        An n-dimensional array of shape (num_samples, ...) is converted into an
        (n+1)-D array of shape (num_frames, window_length, ...), where each frame
        starts hop_length points after the preceding one.
        This is accomplished using stride_tricks, so the original data is not
        copied.  However, there is no zero-padding, so any incomplete frames at the
        end are not included.
        Args:
          data: np.array of dimension N >= 1.
          window_length: Number of samples in each frame.
          hop_length: Advance (in samples) between each window.
        Returns:
          (N+1)-D np.array with as many rows as there are complete frames that can be
          extracted.
        """
        num_samples = data.shape[0]
        num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
        shape = (num_frames, window_length) + data.shape[1:]
        strides = (data.strides[0] * hop_length,) + data.strides
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
