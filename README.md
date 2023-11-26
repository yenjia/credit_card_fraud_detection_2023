# credit_card_fraud_detection_2023
AI CUP 2023 玉山人工智慧公開挑戰賽－信用卡冒用偵測

This method achieved 6th position on the private leaderboard (TEAM_4043).

## Introduction
**There is no new idea in this method**. Just calculated some rule-based features (basic statistics) and classified them by XGBoost classifier.

**Note:** I have not optimized the process. The processing steps are very time-consuming and consume a lot of memory (>50GB). To proceed, ensure you have a good CPU, GPU, and sufficient RAM.

## Preprocessing Data and Checkpoints
The preprocessing data and checkpoints are available in this [Google Drive](https://drive.google.com/drive/folders/1DlS1KMmyNBieRmKBHtb5FlXjPhyk75uE?usp=sharing).
* Preprocessing table: ~14 GB
* XGBoost models: ~400 MB per model

## Data Preprocessing
The data preprocessing process involves concatenating the training, public_test, and private tables to calculate basic statistics for the "cano" and "chid" groups.

**Note:** Before the preprocessing, you should place the tables in the `tables` directory. (The filename of these tables should be `training.csv`, `public.csv` and `private.csv`).
```bash=
cd Preprocessing
python preprocessing.py -o output/preprocessing.csv
```
**Note:** This step could take several hours to complete.

## Training
The model is XGBoost. The file `model.py` contains the parameters which are currently unchangeable.

```
python train.py \
    --input output/preprocessing.csv \
    --model_output_dir  output/checkpoints/ \
    --thr_path config/your_thr.json \
    --epochs 300 \
    --runs 3 \ --> Number of models (for ensemble)
    --gpu 0
```
**Note:** This step could take about 1 hour to complete by using GPU.

## Inference
To perform inference of the data without preprocessing and training your model, download the preprocessing table and model checkpoints first. Then, move them to the `output` directory.

```bash=
python inference_submit.py \
    --input output/preprocessing.csv \
    --thrs config/thr.json \ --> Best thresholds of my models
    --ckpts output/checkpoints/ \
    --output submission.csv
```
**Note:** After inference, you must merge the "txkey" of the example submission file to get the correct submission.