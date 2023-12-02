import argparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

def add_columns(df, keys, csv_set=""):
    """
    Add missing columns to a DataFrame and set a new column "set" to the specified value.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        keys (list[str]): List of column names to add.
        set (str, optional): Value to set for the "set" column. Defaults to "".

    Returns:
        pandas.DataFrame: Updated DataFrame with missing columns added and "set" column set.
    """
    
    print(f"Preparing {csv_set} ...")
    create_col = list(set(keys) - set(df.columns))

    empty_table = pd.DataFrame(columns=create_col)
    df = pd.concat([df, empty_table], axis=1)

    df["set"] = csv_set

    return df[keys]

numerical = [
    "conam",
    "csmam",
    "flam1",
    "iterm",
    "locdt",
    "loctm",
]   

def numerical_preprocessing(data):
    """
    Perform numerical preprocessing on the data.

    Args:
        data (pd.DataFrame): DataFrame containing the data to be preprocessed.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional features.
    """

    for i in tqdm(numerical):
        data[i] = data[i].fillna(-1)
        data[f"normd_group_{i}"] = (data[i] - data.groupby("cano")[i].transform("min")) \
        / (data.groupby("cano")[i].transform("max") - data.groupby("cano")[i].transform("min") + 1e-8)
        data[f"mean_normd_group_{i}"] = data.groupby(["cano"])[f"normd_group_{i}"].transform("mean")
        data[f"diff_normd_group_{i}"] = abs(data[f"normd_group_{i}"] - data[f"mean_normd_group_{i}"])
        
        data[f"normd_group_{i}_2"] = (data[i] - data.groupby("chid")[i].transform("min")) \
        / (data.groupby("chid")[i].transform("max") - data.groupby("chid")[i].transform("min") + 1e-8)
        data[f"mean_normd_group_{i}_2"] = data.groupby(["chid"])[f"normd_group_{i}_2"].transform("mean")
        data[f"diff_normd_group_{i}_2"] = abs(data[f"normd_group_{i}_2"] - data[f"mean_normd_group_{i}_2"])

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="", help="Save the csv")
    args = parser.parse_args()

    # Get the list of required columns
    keys = json.load(open("./numerical_keys.json"))
    ori_keys = json.load(open("./original_keys.json"))

    # Load training and test data
    training = add_columns(pd.read_csv("../tables/training.csv", usecols=ori_keys), keys, "training")
    training = training[training["set"] != "private"]
    public_test = add_columns(pd.read_csv("../tables/private_1.csv"), keys, "public_test")
    private = add_columns(pd.read_csv("../tables/private_2_processed.csv"), keys, "private")

    print("Concat ...")
    data = pd.concat([training, public_test, private], ignore_index=True)

    print("Preprocessing ...")
    # Count the number of "cano" and "txkey" under the group "chid"
    data["sum_group_cano"] = data["chid"].map(data.groupby("chid")["cano"].agg(lambda x: x.nunique()))
    data["sum_group_txkey"] = data["chid"].map(data.groupby("chid")["txkey"].agg(lambda x: x.nunique()))

    print("Numerical columns ...")
    # Preprocess numerical columns
    data = numerical_preprocessing(data)

    # Save the preprocessing data csv
    data.to_csv(args.output, index=None)