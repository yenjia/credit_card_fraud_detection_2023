import argparse
import os
import json
from glob import glob
import numpy as np
import pandas as pd
from Model.model import xgb_model

no_usecols = [
    "txkey",
    "chid",
    "cano",
    "set",
]

def load_preprocessing(input, keys):
    """
    Load the preprocessing table from a CSV file and cast certain columns to categories.

    Args:
        input (str): Path to the CSV file containing the preprocessing table.
        keys (List[str]): List of columns to include in the preprocessing table.

    Returns:
        pd.DataFrame: Preprocessing table with the specified columns and categories.
    """

    preprocessing_table = pd.read_csv(input)
    preprocessing_table = preprocessing_table[keys]
    change_to_categories = [
        "mchno", 
        "acqic", 
        "mode_group_acqic", 
        "mode_group_acqic_2", 
        "mode_group_mchno", 
        "mode_group_mchno_2"
    ]

    for i in change_to_categories:
        preprocessing_table[i] = preprocessing_table[i].astype("category")

    return preprocessing_table



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="output/preprocessing.csv", help="Preprocessing csv")
    parser.add_argument("--thrs", default="config/thr.json", help="Threshold path")
    parser.add_argument("--ckpts", default="output/checkpoints/")
    parser.add_argument("-o", "--output", default="Submitted results")
    args = parser.parse_args()

    # Prepare data
    keys = json.load(open("config/columns_keys.json"))
    thrs = json.load(open(args.thrs))
    preprocessing_table = load_preprocessing(args.input, keys)
    test_df = preprocessing_table[preprocessing_table["set"] != "training"][preprocessing_table.columns.difference(no_usecols)]
    models_list = glob(os.path.join(args.ckpts, "*.json"))
    n_models = len(models_list)

    # Inference
    submit = pd.DataFrame()
    submit["txkey"] = preprocessing_table[preprocessing_table["set"] != "training"]["txkey"].tolist()
    submit["set"] = preprocessing_table[preprocessing_table["set"] != "training"]["set"].tolist()

    for idx, model_path in enumerate(models_list):
        model = xgb_model()
        model.load_model(model_path)
        test_probs = model.predict_proba(test_df.drop("label", axis=1))
        test_preds = (test_probs[:, 1]>thrs[idx]).astype(int)
        submit[f"pred_{idx}"] = test_preds

    # Get the ensemble prediction
    submit["pred"] = submit[[f"pred_{i}" for i in range(n_models)]].sum(axis=1)
    submit["pred"] = (submit["pred"] >= (n_models+1)//2).astype(int)

    # results = pd.read_csv("output/example.csv", usecols=["txkey"])
    # results = results.merge(submit[["txkey", "pred"]], on="txkey")
    submit[["txkey", "pred"]].to_csv(args.output, index=None)
