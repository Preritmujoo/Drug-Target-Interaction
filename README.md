# Drug-Target Interaction Prediction with Neural Networks and Hyperparameter Optimization

## Introduction

Drug-Target Interaction (DTI) prediction is a crucial problem in computational biology, with applications in drug discovery and personalized medicine. This project explores the use of Artificial Neural Networks (ANNs) to predict interactions between drugs and targets based on a synthetic dataset. To enhance model performance, hyperparameter optimization is performed using Keras Tuner's Hyperband algorithm.

## Theoretical Background

1. Drug-Target Interaction (DTI) Prediction
DTI prediction involves identifying the interaction between chemical compounds (drugs) and biological molecules (targets, typically proteins). Predictive models help prioritize experiments by narrowing down promising drug-target pairs. Traditional approaches include:

- Experimental assays: High-cost and time-intensive.
- Computational methods: Leveraging machine learning models for faster and cost-effective predictions.

2. Artificial Neural Networks (ANNs)
ANNs are powerful machine learning models inspired by biological neural networks. Key characteristics:

- Input Layer: Processes raw input features.
- Hidden Layers: Captures complex feature relationships using activation functions.
- Output Layer: Produces predictions (binary classification in this case).

3. Hyperparameter Optimization
Hyperparameters control the learning process (e.g., number of layers, units per layer, and learning rate). Hyperparameter tuning aims to find the optimal combination of these values to improve model performance. Hyperband:

Dynamically allocates resources to promising hyperparameter configurations.
Balances exploration and exploitation in the search space.
Objective
The primary objective of this project is to predict drug-target interactions with high accuracy and optimize the neural network using automated hyperparameter tuning.

## Methodology
1. Data Preprocessing
- Dataset: Synthetic dataset (synthetic_drug_target_interaction.csv) containing drug-target feature pairs and interaction labels.
- Splitting: Data divided into training (80%) and testing (20%) sets.
- Feature Scaling: StandardScaler is applied to normalize the data.

3. Model Architecture
- Input Layer: Tunable units (32–512).
- Hidden Layers: Variable count and configuration.
- Activation Function: ReLU for non-linear transformations.
- Output Layer: Single node with sigmoid activation for binary classification.

5. Hyperparameter Tuning
Key tunable parameters:
- Number of units per layer: Controls the width of the network.
- Number of hidden layers: Controls the network depth.
- Learning rate: Optimizer step size for gradient descent.

6. Training and Evaluation
- Training: Models are trained using the Adam optimizer and evaluated on validation data.
- Metrics: Accuracy and Mean Squared Error (MSE) are used for evaluation.

## Results
1. Optimized Model Performance
- Test Accuracy: Achieved by the best hyperparameter configuration.
- Mean Squared Error (MSE): Quantifies the prediction error.

3. Hyperparameter Insights
Top-performing configurations revealed the optimal layer width, network depth, and learning rate.

5. Visualizations
Validation accuracy vs. number of layers and learning rates.
Directed graph representation of the model architecture.

GitHub-Link:   https://github.com/Preritmujoo/Drug-Target-Interaction
