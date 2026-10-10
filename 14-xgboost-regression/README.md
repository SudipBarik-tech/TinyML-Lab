# Experiment 14 — XGBoost Regression

## Objective
Train a compact gradient-boosted tree regressor and export the learned trees as lightweight C/C++ prediction logic for embedded use.

## Method
A small synthetic regression dataset is generated locally. The XGBoost model is intentionally constrained with shallow trees and a small number of estimators. The training script evaluates regression error and converts every fitted tree into nested C/C++ conditions saved in `embedded/xgb_regressor_model.h`.

## Run
```bash
python 14-xgboost-regression/train.py
```

Then open `embedded/xgb_regression_demo.ino` after the model header is generated.

## TinyML relevance
Boosted trees can provide good nonlinear regression performance without a neural-network runtime. Their embedded cost depends strongly on the number and depth of trees, so model size must be controlled during training.

## Output to record
- MAE and RMSE
- number of trees
- maximum tree depth
- generated header size
- device prediction and latency

## Limitation
The exporter is designed for numeric split trees produced by the configured XGBoost model. More complex objectives or categorical features may require additional handling.
