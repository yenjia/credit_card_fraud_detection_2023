import argparse
import json
import numpy as np
import pandas as pd
from Model.model import xgb_model

no_usecols = [
    "txkey",
    "chid",
    "cano",
    "set",
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", default="output/checkpoints/")
    parser.add_argument("-o", "--output", default="Submitted results")
    args = parser.parse_args()

    keys = json.load(open("config/columns_keys.json"))
    thrs = json.load(open("config/thr.json"))
    preprocessing_table = pd.read_csv("output/preprocessing.csv")
    preprocessing_table = preprocessing_table[keys]
    test_df = preprocessing_table[preprocessing_table["set"] != "training"][preprocessing_table.columns.difference(no_usecols)]
    models_list = args.ckpts
    n_models = len(models_list)

    # Inference
    submit = pd.DataFrame()
    submit["txkey"] = preprocessing_table[preprocessing_table["set"] != "training"]["txkey"].tolist()
    submit["set"] = preprocessing_table[preprocessing_table["set"] != "training"]["set"].tolist()
    
    for idx, i in enumerate(models_list):
        model = xgb_model().load_model(i)
        test_probs = model.predict_proba(test_df.drop("label", axis=1))
        test_preds = (test_probs[:, 1]>thrs[idx]).astype(int)
        submit[f"pred_{i}"] = test_preds

    submit["pred"] = submit[[f"pred_{i}" for i in range(n_models)]].sum(axis=1)
    submit["pred"] = (submit["pred"] >= (n_models+1)//2).astype(int)

    results = pd.read_csv("output/example.csv", usecols=["txkey"])
    results = results.merge(submit[["txkey", "pred"]], on="txkey")
    results.to_csv(args.output, index=None)
