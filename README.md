Machine Learning Project 2:
Income Prediction Model using Multi-Layer Perceptron
A machine learning project that predicts income levels (>50K or ≤50K) using PyTorch neural networks.

Project Overview:
This project implements a Multi-Layer Perceptron (MLP) to classify income levels based on various demographic and employment features. 

Dataset:
project_adult.csv
project_validation_inputs.csv: Validation dataset for predictions
Output: Group_9_MLP_PredictedOutputs.csv

Data preprocessing:
One-hot encoding for categorical variables
Standard scaling for numerical features
Missing value imputation

Model architecture comparison:
Different activation functions (ReLU, Tanh, Sigmoid, Identity)
Various hidden layer sizes (16, 32, 64, 128 units)

Best Model Configuration Using hyperparameter tuning:
Architecture: Single hidden layer with 16 units
Activation Function: ReLU
Learning Rate: 0.01
Optimizer: Adam
Loss Function: Cross Entropy


The final model achieved:

Test Accuracy: ~84.86%
Performance metrics visualized through confusion matrix and detailed metric breakdown


Load and preprocess the data
Train different model configurations
Evaluate and compare performances
Generate predictions for validation set
The final model is saved as income.pt for future use.


  
