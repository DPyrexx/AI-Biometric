
# AI Biometric Recognition Project

## Overview
This project focuses on developing a Machine Learning (ML) solution for biometric recognition with the highest accuracy possible. The primary method used is a traditional Multilayer Perceptron (MLP) model, with extensive experiments conducted to optimize the model's performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Experiments and Results](#experiments-and-results)
4. [Random Search Optimization](#random-search-optimization)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
7. [How to Run the Project](#how-to-run-the-project)
8. [References](#references)

## Introduction
The goal of this project is to develop an ML solution for a biometric recognition task. This involves using MLP models to achieve high recognition accuracy. The project includes:
- Selection of MLP models
- Implementation of K-fold cross-validation to prevent overfitting
- Optimization of parameters through manual and automated methods (Grid Search and Random Search)

## Methodology
The methodology involves several steps:
1. **Data Collection and Processing**: Images and data are preprocessed using custom scripts (`Process_yale_images` and `classify_yale`).
2. **Model Implementation**: Traditional MLP models are used, with a focus on finding the optimal set of parameters through various optimization techniques.
3. **Parameter Optimization**: Initially, parameters were manually adjusted, followed by the use of Grid Search and Random Search to efficiently find the best parameters.

### Key Parameters:
- `hidden_layer_sizes`: Number of neurons in the hidden layers
- `solver`: Optimization algorithm
- `activation`: Activation function in the hidden layers
- `batch_size`: Number of samples per iteration for weight updates
- `max_iter`: Maximum iterations for training
- `learning_rate_init`: Initial learning rate
- `momentum`: Momentum for gradient descent
- `early_stopping`: Early stopping to prevent overfitting
- `validation_fraction`: Proportion of training data used as validation
- `verbose`: Verbosity of training output

## Experiments and Results
Several experiments were conducted to find the optimal parameters:
1. **Changing Hidden Neurons**: Various numbers of hidden neurons were tested.
2. **Changing PCA Components**: Different numbers of PCA components were used to see their effect on accuracy.
3. **K-fold Cross Validation**: Implemented to ensure model robustness and prevent overfitting.

### Results:
- Manual parameter tuning showed initial results but was inefficient.
- Grid Search was exhaustive and computationally expensive.
- Random Search provided an efficient and effective way to optimize parameters, resulting in a best accuracy of 96%.

## Random Search Optimization
Random Search was used to efficiently search through the hyperparameter space. The best parameters found were:
- `hidden_layer_sizes`: 500
- `PCA`: 200
- Other parameters as listed in the methodology.

## Conclusion
This project successfully developed an ML solution for biometric recognition with high accuracy. The use of MLP models and optimization techniques like Random Search significantly improved the model's performance. Future work can focus on further optimization and exploring different ML models.

## Future Work
Future improvements can include:
- Extending the study to include more advanced models and techniques.
- Increasing the dataset size and diversity.
- Implementing real-time biometric recognition systems.

## How to Run the Project
To run this project locally, follow these steps:

### Prerequisites
- Python 3.7 or higher
- Required Python libraries: numpy, scikit-learn, matplotlib, etc.

### Installation
1. Clone the repository:
    \`\`\`bash
    git clone https://github.com/DPyrexx/AI-Biometric.git
    cd AI-Biometric
    \`\`\`

### Data Preparation
Ensure you have the necessary datasets. You may need to download the Yale face dataset or similar biometric datasets and place them in the appropriate directory.

### Running the Scripts
1. **Preprocess the data:**
    \`\`\`bash
    python Process_yale_images.py
    \`\`\`


### Configuration
You can adjust the parameters and settings in the configuration files or directly within the scripts to fit your needs.

## References
For a detailed explanation of the methods, experiments, and results, please refer to the full reports:
- [AI Biometric Report 1]
- [AI Biometric Report 2]
