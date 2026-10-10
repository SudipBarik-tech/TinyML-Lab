# TinyML Lab

Practical experiments in **embedded machine learning, TinyML and resource-efficient AI**.

This repository is maintained as a hands-on academic workspace for studying how machine-learning models move from desktop training environments to constrained embedded devices. The focus is on reproducibility, compact implementations, deployment constraints and measurable embedded performance.

## Scope

The repository is organised into four tracks:

### 1. Embedded ML Fundamentals
Decision Tree, Random Forest, XGBoost, Gaussian Naive Bayes, Support Vector Machine, SEFR, PCA, Logistic Regression, K-Means, Linear Regression, Gaussian Mixture Model, Poisson Regression, KNN, Elastic Net and Huber Regression.

### 2. Tiny Neural Networks
Multilayer Perceptron, CNN, LSTM, Autoencoder and RNN implementations aimed at compact inference workflows.

### 3. Model Optimisation for TinyML
Quantisation-Aware Training, post-training pruning, knowledge distillation and compact MobileNet-style architectures.

### 4. Advanced Edge-AI Experiments
Q-Learning, Kolmogorov-Arnold Networks, LDA, RBF networks, GCNs, Neural Additive Models, Bayesian Neural Networks, Spiking Neural Networks, Liquid Neural Networks, Causal Decision Trees, GANs, ANFIS, Quantile Regression Neural Networks and Self-Organising Maps.

These advanced experiments are included as exploratory edge-AI studies. Not every architecture is naturally suitable for direct deployment on a microcontroller, so deployment feasibility is treated as part of the experiment rather than assumed in advance.

## Experimental workflow

Each experiment should answer the same practical questions:

1. What problem is being solved?
2. What dataset or input representation is used?
3. How is the model trained and evaluated?
4. Can the model be simplified, compressed or exported?
5. Can inference run on a constrained target?
6. What are the memory, latency and accuracy trade-offs?

Where embedded deployment is practical, the experiment will include both desktop-side model development and device-side inference code.

## Target platforms

Primary targets include:

- ESP32 and ESP32-S3
- Arduino-compatible microcontrollers
- TensorFlow Lite for Microcontrollers
- Python and Jupyter for training, testing and model conversion

Hardware and library requirements are documented inside each experiment folder.

## Repository layout

```text
TinyML-Lab/
├── 01-decision-tree/
├── 02-random-forest/
├── 03-xgboost/
├── ...
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

A standard experiment is organised as:

```text
experiment-name/
├── README.md
├── notebook/
├── embedded/
├── data/
└── results/
```

The experiment README should document the objective, method, training process, deployment approach, expected output, measured results, limitations and references.

## Experiment index

| No. | Experiment | Track |
|---:|---|---|
| 01 | Decision Tree | Embedded ML |
| 02 | Random Forest | Embedded ML |
| 03 | XGBoost Classifier | Embedded ML |
| 04 | Gaussian Naive Bayes | Embedded ML |
| 05 | Support Vector Machine | Embedded ML |
| 06 | SEFR Classifier | Embedded ML |
| 07 | Principal Component Analysis | Embedded ML |
| 08 | Logistic Regression | Embedded ML |
| 09 | K-Means | Embedded ML |
| 10 | Linear Regression | Embedded ML |
| 11 | Gaussian Mixture Model | Embedded ML |
| 12 | Multilayer Perceptron | Tiny Neural Network |
| 13 | Convolutional Neural Network | Tiny Neural Network |
| 14 | XGBoost Regression | Embedded ML |
| 15 | Poisson Regression | Embedded ML |
| 16 | K-Nearest Neighbours | Embedded ML |
| 17 | Elastic Net | Embedded ML |
| 18 | LSTM | Tiny Neural Network |
| 19 | Autoencoder | Tiny Neural Network |
| 20 | Q-Learning | Advanced Edge AI |
| 21 | Huber Regression | Embedded ML |
| 22 | Quantisation-Aware Training | TinyML Optimisation |
| 23 | Post-Training Pruning | TinyML Optimisation |
| 24 | Knowledge Distillation | TinyML Optimisation |
| 25 | Micro MobileNet | TinyML Optimisation |
| 26 | Multinomial Naive Bayes | Embedded ML |
| 27 | Kolmogorov-Arnold Network | Advanced Edge AI |
| 28 | Linear Discriminant Analysis | Advanced Edge AI |
| 29 | Radial Basis Function Network | Advanced Edge AI |
| 30 | Graph Convolutional Network | Advanced Edge AI |
| 31 | Neural Additive Model | Advanced Edge AI |
| 32 | Bayesian Neural Network | Advanced Edge AI |
| 33 | Spiking Neural Network | Advanced Edge AI |
| 34 | Recurrent Neural Network | Tiny Neural Network |
| 35 | Liquid Neural Network | Advanced Edge AI |
| 36 | Causal Decision Tree | Advanced Edge AI |
| 37 | Generative Adversarial Network | Advanced Edge AI |
| 38 | ANFIS | Advanced Edge AI |
| 39 | Quantile Regression Neural Network | Advanced Edge AI |
| 40 | Self-Organising Map | Advanced Edge AI |

## Reproducibility

Experiments will be added only after their code, dependencies, file names and documentation have been reviewed. Where possible, results will include accuracy or error measures together with deployment-oriented measurements such as model size, RAM use and inference latency.

## Academic use

The repository is intended for teaching, laboratory practice and reproducible experimentation. Students should use the material to understand the complete workflow and reproduce the experiments rather than submit repository content directly as assessed work.

## Maintainer

**Sudip Barik**  
Assistant Professor, Department of MCA  
Techno International New Town, Kolkata

[GitHub](https://github.com/SudipBarik-tech) · [LinkedIn](https://www.linkedin.com/in/sudip-barik/) · [Google Scholar](https://scholar.google.com/citations?user=qBOJ6w4AAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0002-1284-5575)

## References and attribution

This repository is being rebuilt as an independently maintained academic implementation. External papers, datasets, libraries, code fragments or other third-party material used in an experiment will be credited in that experiment's documentation. Existing third-party license or attribution requirements will be retained where applicable.
