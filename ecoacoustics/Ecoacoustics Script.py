from pathlib import Path
import torch
from conduit.data.datamodules.audio import EcoacousticsDataModule
from sklearn import ensemble, model_selection, tree, metrics
from tqdm import tqdm
import numpy as np


# Initialise a datamodule for testing.
root_dir = Path().resolve().parent.parent
dm = EcoacousticsDataModule(root_dir, specgram_segment_len=1, test_prop=0, val_prop=0, train_batch_size=1)
dm.prepare_data()
dm.setup()

print(f'The train/test/validation split of the dataset is:\n  {dm.num_train_samples}/{dm.num_test_samples}/{dm.num_val_samples}.')

# Get data loader, all data.
data_loader = dm.train_dataloader()

# Define device.
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load VGGish.
model = torch.hub.load('DavidHurst/torchvggish', 'vggish', preprocess=False, device=device, postprocess=False)

# Test loop.
model.eval()
preds = []
labels = []
i = 0
with torch.no_grad():
    for data in tqdm(data_loader, desc="Generating Representations"):
        x = data.x.to(device)
        y = data.y.to(device)

        pred = model(x)
        preds.append(pred.cpu().numpy())
        labels.append(y.cpu().numpy())

        i += 1
        if i > 10_000:
            break

preds = [p.squeeze() for p in preds]
labels = np.asarray(labels).ravel()

# Save memory, del spectrogram encoder.
del model

print(f'Shape of first sample = {preds[0].shape}')
print(f'Label 1 = {labels[0]}, shape = {labels[0].shape}')
            
# Split data into train and test for random forest.
X_train, X_test, y_train, y_test = model_selection.train_test_split(preds, labels, test_size=0.20)

# Fit random forest to data and get predictions.
rm_clf = ensemble.RandomForestClassifier()
rm_clf.fit(X_train, y_train)
y_test_pred = rm_clf.predict(X_test)

# Evaluate classifier's performance.
f1 = metrics.f1_score(y_test, y_test_pred, average='micro')

print(f'The F1 score for the predictions is: {f1:.2f}')
