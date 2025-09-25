# ===================================================================
# PROJECT: Stock Default Prediction Modeling in R
# AUTHOR: Yuxi Wei
#
# DESCRIPTION: This script uses technical indicators to train and
#              evaluate five ML models for predicting stock defaults,
#              defined by a Value at Risk (VaR) threshold.
# ===================================================================


# 1. SETUP: LOAD LIBRARIES
# -------------------------------------------------------------------
# Data manipulation and financial indicators
library("openxlsx")
library("TTR")
library("lubridate")
library("xts")
# Machine learning models
library("e1071") # SVM
library("nnet")  # Neural Networks
library("kknn")  # K-Nearest Neighbors
library("rpart") # Decision Trees
# Model evaluation
library("pROC")


# 2. DATA LOADING AND PREPARATION
# -------------------------------------------------------------------
# Set working directory if the data file is in a different location
# setwd("D:/Desktop") # Uncomment to set your specific path

# Load the stock data from the Excel file
# The original file is expected to have Chinese headers.
df <- read.xlsx("conchcement.xlsx", detectDates = TRUE)

# Rename columns to English for consistency and readability
colnames(df) <- c("Code", "Name", "Date", "Open", "High", "Low", "Close", "Volume", "Turnover", "Change", "Change_Pct", "Amplitude")

# Ensure 'Date' column is in Date format
df$Date <- as.Date(df$Date, '%y%m%d')

# Calculate daily log returns
df$Return <- c(NA, log(df$Close[2:nrow(df)] / df$Close[1:(nrow(df) - 1)]))

# Remove the first row which has an NA return value
df <- df[-1, ]

# Visualize the closing price and daily returns
par(mfrow = c(1, 2))
plot(df$Date, df$Close, type = 'l', col = 'blue', xlab = 'Date', ylab = 'Closing Price')
plot(df$Date, df$Return, type = 'l', col = 'red', xlab = 'Date', ylab = 'Log Return')


# 3. FEATURE ENGINEERING & TARGET VARIABLE CREATION
# -------------------------------------------------------------------
# --- Define the "Default" event using Historical Value at Risk (VaR) ---
alpha <- 0.75 # Set the confidence level for VaR
sorted_returns <- sort(df$Return)
n_returns <- length(sorted_returns)
var_index <- floor(n_returns * (1 - alpha))
dVaR <- -sorted_returns[var_index] # VaR threshold (positive value)

# Create the binary target variable 'Default'
# A "default" is when the negative price change exceeds the VaR threshold.
df$Default <- ifelse(df$Change_Pct < -dVaR, 1, 0)


# --- Calculate Technical Indicators as Features ---

# MACD (Moving Average Convergence Divergence)
macd_data <- MACD(df$Close, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA", percent = FALSE)
df$MACD <- macd_data[, "macd"] - macd_data[, "signal"]

# KDJ (Stochastic Oscillator)
kdj_data <- stoch(df[, c("High", "Low", "Close")], nFastK = 9, nFastD = 3, nSlowD = 3, maType = "SMA")
df$K <- kdj_data[, "fastK"] * 100
df$D <- kdj_data[, "fastD"] * 100
df$J <- 3 * df$K - 2 * df$D

# OBV (On-Balance Volume)
df$OBV <- OBV(df$Close, df$Volume)

# CCI (Commodity Channel Index)
df$CCI <- CCI(df[, c('High', 'Low', 'Close')], n = 20, maType = "SMA")

# --- Final Data Cleaning ---
# Remove any rows with NA values that were generated during indicator calculation
model_data <- na.omit(df)
model_data$Default <- as.factor(model_data$Default)


# 4. DATA PREPROCESSING AND SPLITTING
# -------------------------------------------------------------------
# Normalize the numeric predictor variables
cols_to_scale <- c("Turnover", "Amplitude", "Return", "MACD", "K", "D", "J", "OBV", "CCI")
model_data[cols_to_scale] <- scale(model_data[cols_to_scale])

# Generate training and testing sets (70/30 split)
set.seed(42) # Set seed for reproducibility
n_total <- nrow(model_data)
train_indices <- sort(sample(n_total, n_total * 0.7))
train_set <- model_data[train_indices, ]
test_set <- model_data[-train_indices, ]


# 5. MODEL TRAINING
# -------------------------------------------------------------------
formula <- Default ~ Turnover + Amplitude + Return + MACD + K + D + J + OBV + CCI

# --- Model 1: Logistic Regression ---
logit_model <- nnet::multinom(formula, data = train_set, maxit = 500, trace = FALSE)
pred_logit_class <- predict(logit_model, type = 'class', newdata = test_set)
pred_logit_prob <- predict(logit_model, type = 'prob', newdata = test_set)

# --- Model 2: Support Vector Machine (SVM) ---
svm_model <- e1071::svm(formula, data = train_set, kernel = 'linear', cost = 1e5, scale = TRUE, probability = TRUE)
pred_svm_class <- predict(svm_model, test_set)
pred_svm_prob <- attr(predict(svm_model, test_set, probability = TRUE), "probabilities")[, "1"]

# --- Model 3: Neural Network (NNET) ---
nnet_model <- nnet::nnet(formula, data = train_set, size = 5, decay = 0.01, trace = FALSE)
pred_nnet_class <- predict(nnet_model, test_set, type = "class")
pred_nnet_prob <- predict(nnet_model, test_set, type = "raw")

# --- Model 4: K-Nearest Neighbors (KNN) ---
knn_model <- kknn::kknn(formula, train_set, test_set, k = 7, distance = 2)
pred_knn_class <- fitted(knn_model)
pred_knn_prob <- attr(knn_model, "prob")[, "1"]

# --- Model 5: Decision Tree ---
tree_model <- rpart::rpart(formula, data = train_set)
pred_tree_class <- predict(tree_model, test_set, type = "class")
pred_tree_prob <- predict(tree_model, test_set, type = "prob")[, "1"]


# 6. MODEL EVALUATION
# -------------------------------------------------------------------
# --- Custom NP Metric Function (Type I and Type II Errors) ---
calculate_NP <- function(observed, predicted_class) {
  type1_error <- sum(observed == 0 & predicted_class == 1) / sum(observed == 0) # False Positive Rate
  type2_error <- sum(observed == 1 & predicted_class == 0) / sum(observed == 1) # False Negative Rate
  NP <- (1 - type1_error) / type2_error
  return(data.frame(Type1_Error = type1_error, Type2_Error = type2_error, NP_Ratio = NP))
}

print("--- NP Metrics (Error Analysis) ---")
print("Logistic Regression:"); print(calculate_NP(test_set$Default, pred_logit_class))
print("SVM:"); print(calculate_NP(test_set$Default, pred_svm_class))
print("Neural Network:"); print(calculate_NP(test_set$Default, as.factor(pred_nnet_class)))
print("KNN:"); print(calculate_NP(test_set$Default, pred_knn_class))
print("Decision Tree:"); print(calculate_NP(test_set$Default, pred_tree_class))


# --- ROC Curves and AUC ---
par(mfrow = c(2, 3), mar = c(4, 4, 3, 2)) # Setup plot grid
roc_logit <- roc(test_set$Default, pred_logit_prob)
plot(roc_logit, main = paste("Logit ROC\nAUC =", round(auc(roc_logit), 2)), print.auc = FALSE)

roc_svm <- roc(test_set$Default, pred_svm_prob)
plot(roc_svm, main = paste("SVM ROC\nAUC =", round(auc(roc_svm), 2)), print.auc = FALSE)

roc_nnet <- roc(test_set$Default, as.numeric(pred_nnet_prob))
plot(roc_nnet, main = paste("NNET ROC\nAUC =", round(auc(roc_nnet), 2)), print.auc = FALSE)

roc_knn <- roc(test_set$Default, pred_knn_prob)
plot(roc_knn, main = paste("KNN ROC\nAUC =", round(auc(roc_knn), 2)), print.auc = FALSE)

roc_tree <- roc(test_set$Default, pred_tree_prob)
plot(roc_tree, main = paste("Decision Tree ROC\nAUC =", round(auc(roc_tree), 2)), print.auc = FALSE)


# --- CAP Curves (Cumulative Accuracy Profile) ---
plot_CAP <- function(observed, predicted_prob, model_name) {
  obs_pred <- data.frame(prob = predicted_prob, obs = as.numeric(as.character(observed)))
  obs_pred <- obs_pred[order(obs_pred$prob, decreasing = TRUE), ]
  
  mm <- nrow(obs_pred)
  pro_x <- (1:mm) / mm
  pro_y <- cumsum(obs_pred$obs) / sum(obs_pred$obs)
  
  plot(pro_x, pro_y, type = 'l', main = paste(model_name, "CAP Curve"), 
       xlab = "Cumulative Proportion of Samples", ylab = "Cumulative Proportion of Defaults", col = "blue", lwd = 2)
  abline(0, 1, col = 'red', lty = 2) # Add random model line
  abline(v = sum(obs_pred$obs)/mm, h = 1, col = 'gray', lty = 3) # Add perfect model reference
}

par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
plot_CAP(test_set$Default, pred_logit_prob, "Logistic")
plot_CAP(test_set$Default, pred_svm_prob, "SVM")
plot_CAP(test_set$Default, as.numeric(pred_nnet_prob), "Neural Network")
plot_CAP(test_set$Default, pred_knn_prob, "KNN")
plot_CAP(test_set$Default, pred_tree_prob, "Decision Tree")
