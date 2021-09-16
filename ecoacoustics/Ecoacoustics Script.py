from pathlib import Path
import torch
from conduit.data.datamodules.audio import EcoacousticsDataModule
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm

# Initialise a datamodule for testing.
root_dir = Path().resolve().parent.parent
dm = EcoacousticsDataModule(root_dir, specgram_segment_len=1, test_prop=0, val_prop=0, train_batch_size=1)
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
i = 0
with torch.no_grad():
    for data in tqdm(data_loader, desc="Predicting"):
        x = data.x.to(device)
        y = model(x)
        preds.append(y.cpu().numpy())
        i += 1
        if i > 100:
            break

print(f'Prediction shape: {preds[0].size()}')
print('Values of prediction 0:', preds[0])
np.savetxt(root_dir / "preds.csv", np.asarray(preds), delimiter=",")

class PredictionDataset(Dataset):
    

rm_clf = RandomForestClassifier()


