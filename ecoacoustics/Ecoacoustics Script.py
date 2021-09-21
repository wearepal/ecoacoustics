#!/usr/bin/env python3

"""
    A script to demonstrate the Conduit EcoacousticsDataModule, using it to generate some 
    representations of the audio samples and clasisfying them with a random forest.
"""

from pathlib import Path
import torch
from conduit.data.datamodules.audio import EcoacousticsDataModule
from sklearn import ensemble, model_selection, tree, metrics
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Initialise a datamodule (stores dataloaders).
    root_dir = Path().resolve().parent.parent
    dm = EcoacousticsDataModule(root_dir, specgram_segment_len=1, test_prop=0, val_prop=0, train_batch_size=1, num_freq_bins=1024)
    dm.prepare_data()
    dm.setup()

    print(f'The train/test/validation split of the dataset is: {dm.num_train_samples}/{dm.num_test_samples}/{dm.num_val_samples}.')

    # Get data loader, all data.
    data_loader = dm.train_dataloader()

    # Define device.
    device = torch.device('cpu') #"cuda:1" if torch.cuda.is_available() else "cpu")

    # Load pre-trained VGGish.
    model = torch.hub.load('DavidHurst/torchvggish', 'vggish', preprocess=False, device=device, postprocess=False)

    # Test loop.
    model.eval()
    preds_ls = []
    labels_ls = []
    i = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Generating Representations"):
            x = data.x.to(device)
            y = data.y.to(device)
            pred = model(x)
            preds_ls.append(pred.cpu().numpy())
            labels_ls.append(y.cpu().numpy())
            
            if i > 100:
                break
            i += 1

    preds = np.concatenate(preds_ls)
    labels = np.concatenate(labels_ls)
    
    

    # Save memory, delete spectrogram encoder as it's no longer needed.
    del model

    # Split data into train and test for random forest.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(preds, labels, test_size=0.20)

    # Fit random forest to data and get predictions.
    rf_clf = ensemble.RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)

    # Evaluate classifier's performance.
    f1 = metrics.f1_score(y_test, y_pred, average='micro')
    cm = metrics.confusion_matrix(y_test, y_pred)

    print(f'Classes: {rf_clf.classes_}.')

    print(f'The F1 score for the predictions is: {f1:.2f}%.')
    print(f'Confusion matrix: \n{cm}')

    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(cm).plot(ax=ax)
    fig.savefig(root_dir / "random_forest_cm.png")

if __name__ == '__main__':
    main()
