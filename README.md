# TinyML Lab

Practical experiments in **embedded machine learning, TinyML and resource-efficient AI**.

This repository is a hands-on academic workspace for studying how machine-learning models move from desktop training environments to constrained embedded devices. The focus is on reproducible training, compact model export, device-side inference and measurable deployment trade-offs.

## Tracks

- **Embedded ML Fundamentals** — compact classical ML models and feature-processing methods.
- **Tiny Neural Networks** — small neural architectures for constrained inference.
- **TinyML Optimisation** — quantisation, pruning, distillation and compact architectures.
- **Advanced Edge AI** — exploratory methods whose deployment feasibility is evaluated rather than assumed.

## Implemented experiments

| No. | Experiment | Training | Embedded export |
|---:|---|:---:|:---:|
| 01 | Decision Tree | ✅ | ✅ |
| 02 | Random Forest | ✅ | ✅ |
| 03 | XGBoost Classifier | ✅ | ✅ |
| 04 | Gaussian Naive Bayes | ✅ | ✅ |
| 05 | Linear Support Vector Machine | ✅ | ✅ |

Each of these experiments contains an independently written Python training script, a compact model exporter and an Arduino-compatible inference example.

## Experimental workflow

1. define the problem and input features;
2. prepare a reproducible training dataset;
3. train and evaluate the model in Python;
4. reduce model complexity where required;
5. export the fitted model to lightweight C/C++ logic;
6. verify device-side predictions;
7. record accuracy, memory, latency and practical limitations.

## Target platforms

- ESP32 / ESP32-S3
- Arduino-compatible microcontrollers
- Python for model training and validation
- TensorFlow Lite for Microcontrollers where neural-network deployment is appropriate

## Repository layout

```text
TinyML-Lab/
├── 01-decision-tree/
├── 02-random-forest/
├── 03-xgboost/
├── 04-gaussian-naive-bayes/
├── 05-support-vector-machine/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

A typical implemented experiment contains:

```text
experiment-name/
├── README.md
├── train.py
└── embedded/
    └── *_demo.ino
```

Training scripts generate the compact model headers required by the corresponding embedded sketches.

## Roadmap

### Embedded ML Fundamentals
Decision Tree, Random Forest, XGBoost, Gaussian Naive Bayes, Support Vector Machine, SEFR, PCA, Logistic Regression, K-Means, Linear Regression, Gaussian Mixture Model, Poisson Regression, KNN, Elastic Net, Huber Regression and Multinomial Naive Bayes.

### Tiny Neural Networks
Multilayer Perceptron, CNN, LSTM, Autoencoder and RNN.

### TinyML Optimisation
Quantisation-Aware Training, post-training pruning, knowledge distillation and compact MobileNet-style architectures.

### Advanced Edge-AI Experiments
Q-Learning, Kolmogorov-Arnold Networks, LDA, RBF networks, GCNs, Neural Additive Models, Bayesian Neural Networks, Spiking Neural Networks, Liquid Neural Networks, Causal Decision Trees, GANs, ANFIS, Quantile Regression Neural Networks and Self-Organising Maps.

Not every advanced architecture is naturally suitable for direct deployment on a microcontroller. In those cases, deployment feasibility itself is part of the experiment.

## Reproducibility

The implemented examples use fixed random seeds and explicit model constraints. Generated model headers are produced from the fitted Python models rather than being manually copied into the embedded code.

Install the current Python dependencies with:

```bash
pip install -r requirements.txt
```

Then run an experiment from the repository root, for example:

```bash
python 01-decision-tree/train.py
```

## Academic use

The repository is intended for teaching, laboratory practice and reproducible experimentation. The emphasis is on understanding the complete TinyML workflow rather than using the material as a copy-and-submit solution.

## Maintainer

**Sudip Barik**  
Assistant Professor, Department of MCA  
Techno International New Town, Kolkata

[GitHub](https://github.com/SudipBarik-tech) · [LinkedIn](https://www.linkedin.com/in/sudip-barik/) · [Google Scholar](https://scholar.google.com/citations?user=qBOJ6w4AAAAJ&hl=en) · [ORCID](https://orcid.org/0000-0002-1284-5575)

## References and attribution

This repository is being developed as an independently maintained academic implementation. External papers, datasets, libraries, code fragments or other third-party material used in an experiment will be credited in that experiment's documentation, with existing license and attribution requirements retained where applicable.
