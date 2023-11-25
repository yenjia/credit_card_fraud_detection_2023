import argparse
import os
import json
import numpy as np
import pandas as pd
from Model.model import xgb_model
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

def select_thr_f1(y_true, y_pred, mode="max"):
    """
    Select a threshold based on F1 score.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        mode (str, optional): Selection mode. Can be either "max" or "balanced".
            "max": Selects the threshold that maximizes the F1 score.
            "balanced": Selects the threshold that minimizes the absolute difference
                between recall and precision.

    Returns:
        float: Selected threshold.
    """

    gt_pos = np.sum(y_true)
    gt_neg = np.sum(y_true==0)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    recall = tpr
    precision = (tpr*gt_pos) / (tpr*gt_pos + fpr*gt_neg + 1e-6)
    f1 = 2*recall*precision / (recall + precision + 1e-6)
    
    if mode == "max":
        return thresholds[np.argmax(f1)]
    elif mode == "balanced":
        return thresholds[np.argmin(np.abs(recall[1:] - precision[1:]))]

def load_preprocessing_data(input):
    """
    Load, preproces, and split data for training and testing.

    Args:
        input (str): Path to the input CSV file.

    Returns:
        tuple:
            data (pandas.DataFrame): Complete dataset.
            train_df (pandas.DataFrame): Preprocessed training data.
            test_df (pandas.DataFrame): Preprocessed test data.
    """

    data = pd.read_csv(input)
    keys = json.load(open("./config/columns_keys.json"))
    data = data[keys]

    # Change the type of columns
    change_to_categories = [
        "mchno", 
        "acqic", 
        "mode_group_acqic", 
        "mode_group_acqic_2", 
        "mode_group_mchno", 
        "mode_group_mchno_2"
    ]

    for i in change_to_categories:
        data[i] = data[i].astype("category")

    # Don't use these columns in the training
    no_usecols = [
        "txkey",
        "chid",
        "cano",
        "set",
    ]

    # training, public_test, private
    train_df = data[data["set"] != "private"][data.columns.difference(no_usecols)]
    test_df = data[data["set"] != "training"][data.columns.difference(no_usecols)]

    return data, train_df, test_df

def train(n_estimators=300, seed=0, save_dir=None):
    """
    Train an XGBoost classifier and choose a threshold based on F1 score.

    Args:
        n_estimators (int, optional): Number of estimators. Defaults to 300.
        seed (int, optional): Random seed for splitting data. Defaults to 0.
        save_dir (str, optional): Directory to save the model. Defaults to None.

    Returns:
        tuple:
            model (xgboost.XGBClassifier): Trained XGBoost model.
            thr (float): Selected threshold for binary classification.
    """

    X_train, X_val, y_train, y_val = train_test_split(
        train_df.drop("label", axis=1), 
        train_df["label"], 
        test_size=.1, 
        stratify=train_df["label"], 
        random_state=seed, 
        shuffle=True
    )
    model = xgb_model(n_estimators=n_estimators)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    
    prob = model.predict_proba(X_val)
    thr = select_thr_f1(y_val, prob[:, 1], mode="balanced")

    if save_dir is not None:
        model.save_model(os.path.join(save_dir, "submit_model_{seed}.json"))

    return model, thr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="", help="Preprocessing csv")
    parser.add_argument("--model_output_dir", default=None, help="Model output directory")
    parser.add_argument("--thr_path", default=None, help="Save thr path")
    parser.add_argument("-e", "--epochs", default=300, \
                        help="Number of epochs for a model")
    parser.add_argument("--runs", default=3, help="Number of models (for ensemble)")
    args = parser.parse_args()

    print("Loading and preprocessing the tables ...")
    data, train_df, test_df = load_preprocessing_data(args.input)

    thrs = []
    for i in range(args.runs):
        print(f"Round {i} ...")
        model, thr = train(
            n_estimators=args.epochs, 
            seed=i, 
            save_dir=args.model_output_dir
        )
    
    # save thresholds for inference
    json.dump(thrs, open(args.thr_path, "w"), indent=4)

    