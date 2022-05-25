#!/usr/bin/env python3

"""
    A script to demonstrate the Conduit EcoacousticsDataModule, using it to generate some
    representations of the audio samples and clasisfying them with a random forest.
"""

from pathlib import Path

from data_analysis import apply_umap, run_random_forest
from data_processing_vggish import prepare_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

ROOT_DIR = Path().resolve().parent.parent
LABEL_NAMES = ["UK1", "UK2", "EC1", "EC3", "EC2", "UK3"]


def main() -> None:
    target_attribute = "habitat"
    representations_file = ROOT_DIR / f"representations_{target_attribute}.npz"

    if not Path(representations_file).is_file():
        representations, labels = prepare_data(ROOT_DIR, target_attribute)
        np.savez(representations_file, representation=representations, label=labels)
    else:
        rep_file = np.load(representations_file)
        representations = rep_file["representation"]
        labels = rep_file["label"]

    # Classify representations using a random forest.
    avg_f1, avg_accuracy, cm = run_random_forest(representations, labels)
    print(f"Random forest average F1 score: {avg_f1 * 100:.2f}%.")
    print(f"Random forest average accuracy score: {avg_accuracy * 100:.2f}%.")
    print(f"Confusion Matrix: \n {cm}")

    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax)
    fig.savefig(ROOT_DIR / f"random_forest_{target_attribute}_cm.png")

    # Apply UMAP dimensionality reduction to representations for visualisation.
    fig, _ = apply_umap(representations, labels, LABEL_NAMES)
    fig.savefig(ROOT_DIR / f"umap_visualisation_{target_attribute}.png")


if __name__ == "__main__":
    main()
