# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Simple functions for analysing audio representations
from tqdm import tqdm
from sklearn import ensemble, metrics, model_selection
import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt


def run_random_forest(representations, labels, n_folds=5):
    """
    Apply a random forest classifier to predict the labels given the representations.
    """

    avg_f1 = 0.0
    avg_accuracy = 0.0
    # Average F1 score over n_trails runs.
    for _ in tqdm(range(n_folds), desc="Random Forest Run"):
        # Split data into train and test for random forest.
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            representations, labels, test_size=0.20, stratify=labels
        )

        # Fit random forest to data and get predictions.
        rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
        rf_clf.fit(x_train, y_train)
        y_pred = rf_clf.predict(x_test)

        # Evaluate classifier's performance.
        f1 = metrics.f1_score(y_test, y_pred, average="micro")
        avg_f1 += f1 / n_folds

        accuracy = metrics.accuracy_score(y_test, y_pred)
        avg_accuracy += accuracy / n_folds

    # Generate confusion matrix for last run.
    cm = metrics.confusion_matrix(y_test, y_pred, normalize="all")

    return avg_f1, avg_accuracy, cm


def apply_umap(representations, labels, label_names):
    """Apply UMAP dimensionality reduction to representations to allow visualisation."""
    fig, ax = plt.subplots()
    # Take the umap parameters from the Sethi paper
    mapper = umap.UMAP(
        metric="euclidean", min_dist=0, n_neighbors=50, random_state=0, n_components=2
    ).fit(representations)
    labels_to_plot = [label_names[lab] for lab in labels]
    umap.plot.points(mapper, labels=np.array(labels_to_plot), ax=ax)
    return fig, ax
