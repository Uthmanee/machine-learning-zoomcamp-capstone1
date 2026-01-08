## Depression Prediction from Lifestyle and Health Factors

![Depression Prediction Banner](image/depression.jpg)

This project develops a predictive machine learning model to estimate the likelihood of depression by analyzing an individual’s demographic, socioeconomic, lifestyle, and health-related factors. Key attributes such as age, marital status, education level, income, physical activity, dietary habits, sleep patterns, mental health history, and family history of depression are used to assess an individual’s current health status. In this study, depression is treated as a chronic medical condition due to its long-term impact on both mental and physical health, including increased risk of cardiovascular and other chronic diseases.

## Problem Statement
Depression is a widespread and persistent health condition that significantly affects quality of life and long-term physical health outcomes. Early identification of individuals at risk of depression is essential for timely intervention and effective management. However, traditional screening methods can be limited by accessibility, subjectivity, and resource constraints. There is a need for a data-driven and automated approach that can identify patterns associated with depression using readily available lifestyle, demographic, and health history data. The objective of this project is to develop a machine learning model that predicts depression as a chronic medical condition, supporting early detection, informed clinical decision-making, and preventive healthcare strategies.

## Overview
The project involves data cleaning, exploratory data analysis (EDA), feature importance evaluation, model selection and evaluation. Multiple models were trained and tuned, and the best-performing model was selected using AUC-ROC as the performance metric.

## Dataset

The dataset used in this project is a **synthetic dataset sourced from Kaggle**. It was generated artificially rather than collected from real-world observations. Download the dataset from [here](https://www.kaggle.com/datasets/anthonytherrien/depression-dataset).

⚠️ **Important Notes:**

- The data generation process is not transparent, and the exact methods and criteria used are unknown.
- As a result, the patterns and relationships in the data may not reflect real-world phenomena.
- Findings from this project should **not** be used for real-world decision-making or predictions. This work is intended solely for **educational and experimental purposes**.

💡 **Implications for Modeling:**

- Models trained on this synthetic data are limited by its quality.
- If the synthetic data does not capture meaningful or realistic relationships, models may struggle to generalize.
- In this project, the weak predictive power (AUC-ROC value) observed in the result section reflects these limitations.

## Result
Several models were trained, including **Logistic Regression, Decision Trees, Random Forest, and XGBoost**.  

- **Best Model:** XGBoost  
- **AUC-ROC:** 53.89%  

The AUC-ROC value is only slightly better than random guessing (~50%), indicating **weak predictive power**. This reflects the limitations of the synthetic dataset rather than the models themselves.  

**Note:** Feature importance analysis did not reveal any strongly predictive features. These results are intended for **educational and experimental purposes** and should **not** be used for real-world predictions.

### Installation & Setup
**1. Clone this repository:**
```
git clone https://github.com/Uthmanee/machine-learning-zoomcamp-capstone1.git
```
**2. Install Pipenv (if not already installed)**
```
    pip install pipenv
```
## Run locally (without docker)
**1. Install all dependencies/packages mentioned in the **Pipfile** within the new virtual environment being created.**
```
pipenv install
```
**2. Run the flask server**
```
# Start the virtual environment by running the command below in the root directory
pipenv shell

# Navigate into the script directory after starting the virtual environment
cd script

# Start the flask server by running
python predict.py

# Make a prediction
python predict_test.py