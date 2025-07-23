
# Health Risk Prediction: EDA and KNN Classifier

## Overview

This project aims to predict whether an individual is at high health risk based on three key health metrics: BMI score, blood pressure variation, and activity level index. The analysis involves a comprehensive Exploratory Data Analysis (EDA) to understand the data's characteristics and relationships, followed by the implementation and evaluation of a K-Nearest Neighbors (KNN) classification model.

The project demonstrates a standard data science workflow, including data inspection, visualization, model training, and performance evaluation using metrics like accuracy, a confusion matrix, and a classification report.

## Dataset

The dataset used in this project is `health_risk.csv`, which contains 1000 records of anonymized patient data. It is a binary classification problem where the goal is to predict the `high_risk_flag`.

### Features

The dataset includes the following features:

| Feature                    | Description                                         | Data Type |
|----------------------------|-----------------------------------------------------|-----------|
| `bmi_score`                | A normalized score representing the Body Mass Index.  | float64   |
| `blood_pressure_variation` | A normalized score for blood pressure variation.      | float64   |
| `activity_level_index`     | A normalized index representing physical activity.    | float64   |
| **`high_risk_flag`**       | **(Target)** The binary flag for health risk (1: Risky, 0: No-Risk). | int64     |

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)

The initial analysis focused on understanding the dataset's structure and uncovering patterns. Key findings from the EDA include:

*   **No Missing Values**: The dataset is clean, with no missing values in any of the columns.
*   **Balanced Classes**: The target variable `high_risk_flag` is well-balanced, with 498 samples labeled as "Risky" (1) and 502 as "No-Risk" (0). This ensures that the model will not be biased towards one class.
*   **Data is Pre-scaled**: As observed from the `df.describe()` output and visualizations, the feature values are already centered around zero and appear to be scaled. This means that an additional scaling step (like `StandardScaler`) is not required before training the KNN model.

Several visualizations were created to explore the data:

*   **Correlation Heatmap**: A heatmap was generated to visualize the correlation between the features. It showed a weak correlation between the predictor variables, indicating that they provide independent information.

    

*   **Scatter Plot**: A scatter plot of `blood_pressure_variation` vs. `activity_level_index` was used to visualize the class separation. The plot indicated that the two classes are reasonably well-separated, making them suitable for classification algorithms.

    

*   **Boxenplot**: A boxenplot was used to show the distribution of each feature. This visualization confirmed the scaled nature of the data and provided insights into the spread and central tendency of each variable.

    

### 2. Model Training and Evaluation

For this classification task, the **K-Nearest Neighbors (KNN)** algorithm was chosen. KNN is a non-parametric, instance-based learning algorithm that classifies a data point based on the majority class of its 'k' nearest neighbors.

**Implementation Steps:**

1.  **Feature and Target Split**: The dataset was divided into features (`X`) and the target variable (`y`).
2.  **Train-Test Split**: The data was split into a 75% training set and a 25% testing set using `train_test_split`. A `random_state` was set for reproducibility.
3.  **Model Training**: A `KNeighborsClassifier` was instantiated with the following parameters:
    *   `n_neighbors=3`: The model considers the 3 nearest neighbors to make a prediction.
    *   `weights='distance'`: Closer neighbors have a greater influence on the prediction than more distant ones.
    The model was then trained on the training data.

### 3. Results

The trained KNN model was evaluated on the unseen test data, and it performed exceptionally well.

*   **Accuracy Score**: The model achieved an accuracy of **96.4%**.

*   **Classification Report**: The detailed report confirmed strong performance for both classes.
    ```
                  precision    recall  f1-score   support

               0       0.95      0.98      0.96       126
               1       0.98      0.94      0.96       124

        accuracy                           0.96       250
       macro avg       0.96      0.96      0.96       250
    weighted avg       0.96      0.96      0.96       250
    ```

*   **Confusion Matrix**: The confusion matrix visually represents the model's performance, showing the number of correct and incorrect predictions for each class. The model made very few errors.
