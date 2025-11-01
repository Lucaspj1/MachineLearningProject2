# Machine Learning Project 2:
## Income Prediction Model using Multi-Layer Perceptron
A machine learning project that predicts income levels (>50K or ≤50K) using PyTorch neural networks.

## Project Overview:
This project implements a Multi-Layer Perceptron (MLP) to classify income levels based on various demographic and employment features. 

## Dataset:
project_adult.csv
project_validation_inputs.csv: Validation dataset for predictions
Output: Group_9_MLP_PredictedOutputs.csv

## Data preprocessing:
One-hot encoding for categorical variables
Standard scaling for numerical features
Missing value imputation

## Model architecture comparison:
Different activation functions (ReLU, Tanh, Sigmoid, Identity)
Various hidden layer sizes (16, 32, 64, 128 units)

## Final Model
### Best Model Configuration Using hyperparameter tuning:
Architecture: Single hidden layer with 16 units
Activation Function: ReLU
Learning Rate: 0.01
Optimizer: Adam
Loss Function: Cross Entropy


## The final model achieved:
Test Accuracy: ~84.86%
Performance metrics visualized through confusion matrix and detailed metric breakdown

## Process
Load and preprocess the data
Train different model configurations
Evaluate and compare performances
Generate predictions for validation set
The final model is saved as income.pt for future use.

## Questions
### Why did you choose the specific architecture (e.g., number of layers, activation functions) for each model?

The hyperparameter tuning we did, tested every single possible numbers of layers and activation functions for each model to fnd the most optimal version of the model.

### How did you monitor and mitigate overfitting in your models?

Model Architecture:

We selected ReLU over tanh for the hidden activation function. ReLU had better testing performance and a small train-testing gap. Tanh overfitted with high training accuracy and poor testing accuracy. ReLU had the smallest gap between training and testing accuracy. For network size, we chose 16 neurons in the hidden layer as the simplest model. Fewer neurons means less memorization capacity. This forces the model to learn patterns to generalize the data. For epochs trained, we chose to train on 100 epochs. More epochs would lead to the model memorizing the training data.

Monitoring Techniques:

We plotted training vs testing loss curves over epochs. We tracked train-test accuracy difference for the accuracy gap. We evaluated multiple configurations through cross-model comparison. We prioritized testing performance over training performance.

### What ethical concerns might arise from deploying models trained on these datasets?

Usage Matters:

When working with data that has social implications as this data does, we need to be careful on how it is used. If this were to be used to maybe accept/deny people for something, it would be unethical to strictly use this model, and it should only be used as an assistance tool.

Sensitive Information:

This dataset contains personal information about individuals. Using people's race, marital status, gender, and more can lead to unfair patterns and repeat past inequalities that existed and still exist.

Privacy:

We need to make sure people's privacy is respected and their information cannot be traced back to them.

Dealing with Missing and Standardized Data:

As nearly all datasets do, this dataset had missing values. By replacing missing data, we risk oversimplification of people and train on data that we are not actually sure of. To produce a successful model, we need to standardize the data and one hot encode variables. This leads to people's characteristics being simplified to 0's and 1's, which is not always a good indication of people's actual information and could affect certain groups of people more than others.

### Why are activation functions necessary in neural networks?

They are necessary for adding Nonlinearity to the model​. This is the most important reason for having activation functions in neural networks​, so that 
we can convert simple linear boundaries into a more complex nonlinear boundary​. It improves dealing with complex data that does not follow a simple linear line​,
allows it to understand complicated patterns within the data , and allows the neurons to turn on and off​. Activation functions allow for the neurons to activate 
and deactivate based on the set parameters and learned weights, to allow for better pattern recognition​.
It also always for Gradient tracking​ to take place. The activation functions are differentiable so that we can calculate the gradients and update weights​



  
