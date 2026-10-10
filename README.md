# TinyML Lab

A practical repository for studying **Tiny Machine Learning, embedded inference and edge AI** on resource-constrained devices.

This repository is being organised as a hands-on learning and experimentation space maintained by **Sudip Barik**. The aim is not only to train machine-learning models, but also to understand how those models can be simplified, converted and deployed on microcontrollers and embedded platforms.

## What this repository covers

The experiments are grouped around four themes:

- classical machine-learning models for embedded inference
- dimensionality reduction and lightweight feature processing
- neural networks for TinyML
- model compression, optimisation and deployment

## Experiment roadmap

| No. | Experiment | Focus |
|---:|---|---|
| 01 | Decision Tree | lightweight rule-based classification |
| 02 | Random Forest | ensemble classification and regression |
| 03 | XGBoost Classifier | boosted-tree inference |
| 04 | Gaussian Naive Bayes | probabilistic classification |
| 05 | Support Vector Machine | margin-based classification |
| 06 | SEFR Classifier | compact binary classification |
| 07 | Principal Component Analysis | feature reduction |
| 08 | Logistic Regression | compact linear classification |
| 09 | K-Means | unsupervised clustering |
| 10 | Linear Regression | numerical prediction |
| 11 | Gaussian Mixture Model | probabilistic clustering |
| 12 | Multilayer Perceptron | basic neural inference |
| 13 | Convolutional Neural Network | compact spatial feature learning |
| 14 | XGBoost Regression | boosted-tree regression |
| 15 | Poisson Regression | count-data modelling |
| 16 | K-Nearest Neighbours | instance-based classification |
| 17 | Elastic Net | regularised regression |
| 18 | LSTM | sequence modelling |
| 19 | Autoencoder | representation learning |
| 20 | Q-Learning | lightweight reinforcement learning |
| 21 | Huber Regression | robust regression |
| 22 | Quantisation-Aware Training | low-precision neural inference |
| 23 | Post-Training Pruning | reducing model complexity |
| 24 | Knowledge Distillation | transferring knowledge to smaller models |
| 25 | Micro MobileNet | compact convolutional architecture |
| 26 | Multinomial Naive Bayes | lightweight text classification |
| 27 | Kolmogorov-Arnold Network | compact nonlinear modelling |
| 28 | Linear Discriminant Analysis | dimensionality reduction and classification |
| 29 | Radial Basis Function Network | nonlinear function approximation |
| 30 | Graph Convolutional Network | graph-based learning |
| 31 | Neural Additive Model | interpretable neural modelling |
| 32 | Bayesian Neural Network | uncertainty-aware inference |
| 33 | Spiking Neural Network | event-driven neural computation |
| 34 | Recurrent Neural Network | sequential inference |
| 35 | Liquid Neural Network | adaptive temporal modelling |
| 36 | Causal Decision Tree | causal structure and decision learning |
| 37 | Generative Adversarial Network | compact generative modelling |
| 38 | ANFIS | neuro-fuzzy inference |
| 39 | Quantile Regression Neural Network | uncertainty and interval estimation |
| 40 | Self-Organising Map | unsupervised topology learning |

## Typical workflow

Each experiment is being structured around the same practical pipeline:

1. prepare or generate the dataset;
2. train and evaluate the model in Python;
3. simplify or convert the trained model where required;
4. implement or export the inference logic for the target device;
5. verify embedded predictions against the desktop model;
6. record memory, latency and deployment observations.

## Target platforms

The repository is intended for experiments using platforms such as:

- ESP32 / ESP32-S3
- Arduino-compatible microcontrollers
- TensorFlow Lite for Microcontrollers
- Python / Jupyter for training and validation

Hardware requirements vary between experiments. Individual experiment folders will contain their own setup notes.

## Repository structure

```text
TinyML-main/
├── 01_decision_tree/
├── 02_random_forest/
├── 03_xgboost/
├── ...
├── 40_self_organising_map/
├── docs/
└── README.md
```

The internal structure will be kept consistent wherever possible:

```text
experiment_name/
├── README.md          # objective, method, steps and observations
├── notebook/          # model development / training
├── embedded/          # microcontroller inference code
├── data/              # small reusable datasets where appropriate
└── results/           # plots, screenshots or measured outputs
```

## Status

This repository is under active restructuring. Experiments are being reviewed one by one before they are added so that the code, documentation, naming and explanations remain consistent across the repository.

## Academic use

The material is intended for learning, laboratory work and reproducible experimentation in **TinyML, IoT and embedded AI**. Students are encouraged to understand the workflow and reproduce the results rather than use the repository as a copy-and-submit solution.

## Maintainer

**Sudip Barik**  
Assistant Professor, Department of MCA  
Techno International New Town, Kolkata

[GitHub](https://github.com/SudipBarik-tech) · [LinkedIn](https://www.linkedin.com/in/sudip-barik/) · [Google Scholar](https://scholar.google.com/citations?user=qBOJ6w4AAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0002-1284-5575)

## Source and attribution note

Some topics in this learning roadmap were inspired by publicly available TinyML examples and academic resources. Any third-party code, datasets or assets retained in an experiment will be identified in that experiment's documentation. The repository README and new implementations are being written and maintained independently for this project.
