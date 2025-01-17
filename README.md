# Mushroom Classification with a Custom Decision Tree Classifier

This project focuses on developing a custom binary decision tree classifier to predict whether a mushroom is edible or poisonous. The project spans several stages, including dataset exploration, preprocessing, classifier construction, hyperparameter tuning, and model evaluation.

## Project Overview

### Dataset Exploration and Preprocessing
The dataset was analyzed through Exploratory Data Analysis (EDA) to understand its structure and features. Preprocessing steps were applied to prepare the data for training, including handling missing values and encoding categorical features. The data was split into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution.

### Custom Decision Tree Classifier
The classifier was implemented by defining two main components:
1. **Node Class**: Represents both decision nodes (splitting points) and leaf nodes (predictions).
2. **Decision Tree Class**: Constructs the decision tree using a recursive algorithm that evaluates potential splits based on feature thresholds. Splits were optimized using criteria such as:
   - Scaled entropy
   - Gini index
   - Squared impurity

Each decision node represents a feature and its threshold, while leaf nodes hold the predicted class label (edible or poisonous).

### Hyperparameter Optimization
Hyperparameter tuning was performed using random search combined with stratified k-fold cross-validation to ensure robust and unbiased evaluation. Optimized parameters included:
- Maximum depth of the tree
- Minimum samples required to split a node
- Criterion for evaluating splits
- Minimum impurity decrease for tree growth stopping

### Model Evaluation
The model's performance was assessed using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

These metrics were calculated on both training and test sets, offering a comprehensive view of the model's ability to generalize and correctly classify mushrooms.

## Key Features
- **Custom Decision Tree Implementation**: Built from scratch, demonstrating an in-depth understanding of tree-based algorithms.
- **Hyperparameter Optimization**: Ensures optimal performance using random search and stratified k-fold cross-validation.
- **Comprehensive Evaluation**: Provides detailed insights into the classifier's robustness and generalization.

## Repository Structure
- `data/`: Contains the following files:
  - `X_train.pkl`, `X_test.pkl`: Pickled training and testing features.
  - `y_train.pkl`, `y_test.pkl`: Pickled training and testing labels.
  - `best_parameters.pkl`: Stores the best hyperparameters identified during tuning.
  - `secondary_data.csv`: The original mushroom dataset.
- `tuning.ipynb`: Jupyter notebook for hyperparameter tuning using random search and stratified k-fold cross-validation.
- `main.ipynb`: Jupyter notebook for data preprocessing, decision tree construction, and model evaluation.
- `Statistical_methods_for_Machine_learning REPORT.pdf`: Detailed report documenting the project's methodology, results, and findings.

