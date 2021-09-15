from pathlib import Path
import torch
from conduit.data.datamodules.audio import EcoacousticsDataModule
import numpy as np
from tqdm import tqdm

# Initialise a datamodule for testing.
root_dir = Path().resolve().parent
dm = EcoacousticsDataModule(root_dir, specgram_segment_len=1, test_prop=0, val_prop=0)
dm.prepare_data()
dm.setup()

print(f'The train/test/validation split of the dataset is:\n  {dm.num_train_samples}/{dm.num_test_samples}/{dm.num_val_samples}.')

# Get data loader, all data.
data_loader = dm.train_dataloader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model.
model = torch.hub.load('DavidHurst/torchvggish', 'vggish', preprocess=False, device=device, postprocess=False)

# Test loop.
model.eval()
preds = []
for data in tqdm(enumerate(data_loader)):
    x = data.x.to(device)
    y = model(x)
    preds.append(y.detach.cpu().numpy())

preds = np.concat(preds, axis=0)

print(len(preds))
print(preds[0])

