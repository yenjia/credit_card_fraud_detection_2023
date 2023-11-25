# Model Description

This method utilizes Focal Loss as the objective function to train an XGBoost classifier. Focal Loss addresses the issue of class imbalance during training by dynamically adjusting the loss based on the confidence of the model's predictions. This approach helps to focus the training process on harder-to-classify examples, leading to improved performance.

## Focal Loss

Settings:
```python
f = FocalLoss(alpha=0.8, gamma=3)
```
This line initializes a FocalLoss object with the specified hyperparameters:

- `alpha`: Control the relative weight given to easy examples.
- `gamma`: Determine the degree to which the loss is reduced for well-classified examples. Higher gamma values emphasize harder examples more.

## XGBoost Classifier

```python
model = XGBClassifier(
    objective=f.focal_binary_object,
    tree_method="hist",
    n_estimators=n_estimators,
    learning_rate=0.5,
    max_depth=12,
    subsample=0.1,
    sampling_method="gradient_based",
    colsample_bytree=1,
    scale_pos_weight=1,
    enable_categorical=True,
    device=device,
    verbosity=1,
    eval_metric=f1_eval,
    importance_type="cover",
    radom_state=0,
)
```

This defines and initializes an XGBoost classifier with the following important parameters:

- `objective`: Specify the FocalLoss object as the objective function.
- `tree_method`: Set the tree construction method to "hist" for histogram-based tree construction.
- `max_depth`: Limit the maximum depth of each decision tree.
- `subsample`: Indicate the fraction of training data used to build each tree. Use a smaller value suggested by documentation.
- `sampling_method`: Choose the sampling strategy for building trees. "gradient_based" emphasizes instances that contribute most to the gradient.
- `eval_metric`: Define the evaluation metric used to assess the model's performance. In this case, F1 evaluation is used.
- `importance_type`: Specify the method for calculating feature importances. "cover" measures the percentage of data points covered by splits on a feature. This parameter increases the performance significantly. 