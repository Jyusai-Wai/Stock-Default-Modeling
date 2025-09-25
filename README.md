# Stock Default Prediction Modeling in R

This R project leverages financial technical indicators to predict significant negative price movements, defined as "default events," for a given stock. The script builds and evaluates five different machine learning classification models to assess their effectiveness in this prediction task.

The analysis was originally performed on historical stock data for Conch Cement, loaded from `conchcement.xlsx`.

## Project Workflow

1.  **Data Loading & Preparation**: The script loads historical stock data from an Excel file, formats dates, and calculates daily log returns.
2.  **Feature Engineering**: A suite of common technical indicators are calculated from the raw price data to serve as predictive features for the models:
    * MACD (Moving Average Convergence Divergence)
    * KDJ (Stochastic Oscillator)
    * OBV (On-Balance Volume)
    * CCI (Commodity Channel Index)
3.  **Target Variable Definition**: A "default event" (the target variable `Default`) is defined using the **Historical Value at Risk (VaR)** method. If a day's negative return exceeds the calculated VaR threshold, it is labeled as a default (`1`); otherwise, it is not (`0`).
4.  **Model Training**: The dataset is scaled and split into a training set (70%) and a testing set (30%). Five distinct classification models are then trained on the engineered features:
    * Logistic Regression
    * Support Vector Machine (SVM)
    * Neural Network (NNET)
    * K-Nearest Neighbors (KNN)
    * Decision Tree (rpart)
5.  **Model Evaluation**: The performance of each model is thoroughly assessed using several standard metrics and visualizations:
    * **ROC Curves** and AUC values to evaluate the trade-off between true positive and false positive rates.
    * **CAP Curves** (Cumulative Accuracy Profile) to measure each model's discriminatory power.
    * A custom **NP Metric** to analyze Type I and Type II error rates.

## How to Use This Project

1.  **Install Required R Packages**:
    Open your R or RStudio console and run the following command to install all necessary packages:
    ```R
    install.packages(c("openxlsx", "TTR", "lubridate", "xts", "e1071", "nnet", "kknn", "rpart", "pROC"))
    ```

2.  **Prepare the Data File**:
    Place your stock data file, named `conchcement.xlsx`, in the same directory as the R script. The Excel file must contain columns named `日期` (Date), `收盘价` (Close), `最高` (High), `最低` (Low), and `成交额` (Volume).

3.  **Run the Analysis Script**:
    Open the `stock_default_modeling.R` script in RStudio and execute it. The script will run the entire analysis pipeline, print evaluation metrics to the console, and generate plots.

## Expected Output

The script will produce a series of plots in the RStudio "Plots" pane, including:
* A chart of the daily closing price and log returns.
* ROC curves for each of the five models, with AUC values displayed.
* CAP curves for each of the five models.
