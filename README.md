# credit_card_fraud_detection_2023
AI CUP 2023 玉山人工智慧公開挑戰賽－信用卡冒用偵測

This method achieved 7th position on the private leaderboard (TEAM_4043).

Final competition slide [[link](https://docs.google.com/presentation/d/1VdH3zyurjtmtQTlQ-H9O0Q6yRWlqe6c4/edit?usp=drive_link&ouid=103785507326024132087&rtpof=true&sd=true)]

## Introduction
**There is no new idea in this method**. Just calculated some rule-based features (basic statistics) and classified them by XGBoost classifier.

**Note:** I have not optimized the process. The processing steps are very time-consuming and consume a lot of memory (>50GB). To proceed, ensure you have a good CPU, GPU, and sufficient RAM.

## Preprocessing Data and Checkpoints
The preprocessing data and checkpoints are not available now. If you need the checkpoints, please contact me.
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

## Final Competition (2023.12.02)
All the code can be found in the `final_code` directory

**Step 1:** Preprocessing. Use `training.csv` provided before the final competition, `private_1.csv` and `private_2_processed.csv` to calculate basic statistics for the "cano" and "chid" groups as features. These files should be placed in the `tables` directory.

However, the CPU performance is insufficient to complete all the preprocessing. So we only transform some essential features (Only including the transformed features of numerical features).

```bash=
cd final_code

python preprocess_numerical.py -o ../output/preprocessing_final.csv
```

**Step2:** Since the environment of the final competition is no GPU, we change the device of XGBoost classifier to "cpu". Also, you must first turn off the parameters `subsample` and `sampling_method` in `model.py` because they are only available on GPU.


```
python train_numerical.py \
    --input ../output/preprocessing_final.csv \
    --model_output_dir  ../output/checkpoints/ \
    --thr_path thr_final.json \
    --epochs 100 \
    --runs 3 \
    --gpu cpu
```

**Step3:** Inference (ensemble 3 models)
```bash=
python inference_numerical.py \
    --input ../output/preprocessing_final.csv \
    --thrs thr_final.json \
    --ckpts ../output/checkpoints/ \
    --output submission.csv
```

**Note:** After step 3, you must merge the "txkey" of the example submission file to get the correct submission.
