# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Prepare our data for data processing with VGGish
import numpy as np
import torch
from conduit.data.datamodules.audio import EcoacousticsDataModule
import torchaudio.transforms as T
from tqdm import tqdm


def prepare_data(root_directory, target_attribute):
    resample_rate = 16_000  # Matching sample rate used by Sethi et al.: https://www.pnas.org/content/117/29/17049.

    # Values following those delineated in https://arxiv.org/pdf/1609.09430.pdf.
    window_length_secs = 0.025
    hop_length_secs = 0.01
    window_length_samples = int(round(resample_rate * window_length_secs))
    hop_length_samples = int(round(resample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    n_freq_bins = 64

    # Define the spectrogram transform we want to apply to the waveform segments.
    specgram_tform = T.MelSpectrogram(
        sample_rate=resample_rate,
        n_mels=n_freq_bins,
        n_fft=fft_length,
        win_length=window_length_samples,
        hop_length=hop_length_samples,
    )

    # Initialise Conduit datamodule (stores dataloaders).
    dm = EcoacousticsDataModule(
        root=root_directory,
        specgram_segment_len=0.96 * 60.0,
        test_prop=0,
        val_prop=0,
        train_batch_size=1,
        target_attr=target_attribute,
        preprocessing_transform=specgram_tform,
        resample_rate=resample_rate,
    )
    dm.prepare_data()
    dm.setup()

    # Pass all data through VGGish.
    representations, labels = process_vggish(dm)

    return representations, labels


def process_vggish(datamodule):
    """Pass data through VGGish to obtain 128 dimensional vector representations of samples."""

    # Get data loader.
    data_loader = datamodule.train_dataloader()

    # Define device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained VGGish.
    model = torch.hub.load(
        "DavidHurst/torchvggish",
        "vggish",
        preprocess=False,
        device=device,
        postprocess=False,
    )

    # Pass all samples through VGGish, storing output representation and correspondig label.
    model.eval()
    reps_ls = []
    labels_ls = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Generating Representations"):
            # Crop spectrogram to meet VGGish's expected dimensions.
            specgram = data.x[:, :, :, : 96 * 60]

            # Transpose spectrogram to get time on the vertical axis instead of freq bins.
            specgram = torch.transpose(specgram, -2, -1)

            specgram = torch.reshape(specgram, (60, 1, 96, 64))

            # VGGish expects log-transformed spectrograms.
            specgram = torch.log(specgram + 10e-4).to(device)
            y = data.y.to(device)

            rep = model(specgram)
            rep = torch.mean(rep, 0, keepdim=True)

            reps_ls.append(rep.cpu().numpy())
            labels_ls.append(y.cpu().numpy())

    reps = np.concatenate(reps_ls)
    labels = np.concatenate(labels_ls)

    # Save memory, delete spectrogram encoder as it's no longer needed.
    del model

    return reps, labels
