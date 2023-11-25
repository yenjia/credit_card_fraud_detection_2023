# credit_card_fraud_detection_2023
AI CUP 2023 玉山人工智慧公開挑戰賽－信用卡冒用偵測

This method achieved 6th position on the private leaderboard (TEAM_4043).

## Introduction
**No new idea in this method**. I just calculated some rule-based features (basic statistics) and classified by XGBoost classifier.

**Note:** I have not optimized the process, the processing steps are very time-consuming and consume a lot of memory (>50GB), it is recommended to have a good CPU, GPU and sufficient RAM to proceed.

## Preprocessing Data and Checkpoints
The preprocessing data and checkpoints are available in this [Google Drive](https://drive.google.com/drive/folders/1DlS1KMmyNBieRmKBHtb5FlXjPhyk75uE?usp=sharing).

## Data Preprocessing
In the data preprocessing process, the tables (training, public_test and private) are concatenated for calculating the basic statistics under the groups of "cano" and "chid".

**Note:** Before the preprocessing, you should place the tables in the `tables` directory. (The filename of these tables should be `training.csv`, `public.csv` and `private.csv`).
```bash=
cd Preprocessing
python preprocessing.py -o output/preprocessing.csv
```
**Note:** This step could take several hours to complete.

## Training
The model is XGBoost. The parameters are currently not changeable and are placed in `model.py`.
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
To infer the data without preprocessing and training your own model, first download the preprocessing table and model checkpoints. Then, place them in the `output` directory.

```bash=
python inference_submit.py \
    --input output/preprocessing.csv \
    --thrs config/thr.json \ --> Best thresholds of my models
    --ckpts output/checkpoints/ \
    --output submission.csv
```
**Note:** After inference, you need to merge the "txkey" of the example submission file to get the correct submission.