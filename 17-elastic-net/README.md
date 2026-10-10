# 17 — Elastic Net Regression

This experiment demonstrates a regularised linear regression model that combines L1 and L2 penalties.

## Objective
Study how Elastic Net can produce a compact regression model suitable for direct embedded inference.

## Method
A reproducible synthetic regression dataset is generated and standardised. `ElasticNet` is trained with fixed regularisation settings. The fitted coefficients, intercept, mean and scale are exported to `embedded/elastic_net_model.h`.

## Run
```bash
python 17-elastic-net/train.py
```

The script reports MAE and RMSE on the test set and generates the embedded header.

## Embedded inference
`embedded/elastic_net_demo.ino` applies the same standardisation used during training and evaluates one dot product plus the intercept.

## TinyML note
Elastic Net is attractive for constrained devices because inference cost is linear in the number of input features and sparsity can reduce the number of effective multiplications.
