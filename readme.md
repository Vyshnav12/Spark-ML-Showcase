# Cluster-Based Data Analysis and Machine Learning

This project consists of several Spark-based analyses, including particle physics classification, web log analysis and collaborative filtering. The code is designed to run on the Sheffield Advanced Research Computer (ShaRC) cluster but can be adapted to other Spark environments.

## Project Structure

The project is divided into the following tasks:

## T1: Model Evaluation and Parameter Tuning for Higgs Particle Classification

**Objective:** Classify Higgs particles using advanced machine learning models and evaluate their performance through parameter tuning and cross-validation.

### Methods Used

- **Model Training and Evaluation:**
  - **Random Forest (RF):** Trained a Random Forest classifier using different hyperparameters. Evaluated the model's performance using accuracy, AUC (Area Under the Curve), precision, recall, and F1 score. Logged feature importances and confusion matrices.
  - **Gradient Boosted Trees (GBT):** Trained a GBT classifier, similarly evaluated using various metrics, and tuned hyperparameters through cross-validation.

- **Parameter Tuning:**
  - **Cross-Validation:** Used a 3-fold cross-validation strategy with parameter grids to identify the best model configurations for both Random Forest and GBT classifiers.
  - **Hyperparameters:** Tuned key hyperparameters including the number of trees, feature subset strategy, subsampling rate (for Random Forest), and max iterations, max depth, subsampling rate (for GBT).

- **Logging and Metrics:**
  - **Confusion Matrix:** Saved confusion matrices to analyze the models' performance in terms of true positives, true negatives, false positives, and false negatives.
  - **Model Metrics:** Logged precision, recall, and F1 score to assess the classification quality.
  - **Training Time:** Recorded the training time for each model to compare computational efficiency.
  - **Feature Importances:** Logged the importance of features as determined by the Random Forest model to understand which features contributed most to the predictions.
  - **Model Parameters:** Documented the best hyperparameters for both classifiers.

### Outputs

- **Confusion Matrices:** Confusion matrices for Random Forest and GBT models, saved as `RandomForest_confusion_matrix.txt` and `GBT_confusion_matrix.txt`.
- **Model Metrics:** Precision, recall, and F1 scores for each model, saved as `RandomForest_metrics.txt` and `GBT_metrics.txt`.
- **Training Times:** The training times for each model are logged in `training_times.txt`.
- **Feature Importances:** The feature importances for the Random Forest model, saved as `RandomForest_feature_importances.txt`.
- **Model Parameters:** The best model parameters for Random Forest and GBT, saved in `model_parameters.txt`.

## T2: Insurance Claims Prediction

**Objective:** Predict the occurrence of claims using machine learning models with regularization and cross-validation.

### Methods Used

- **Data Preparation:**
  - Loaded and preprocessed the dataset, converting numeric columns and creating new features (`NZClaim`, `PP_ClaimNb`, `LogClaimNb`).
  - Performed a stratified 70/30 train-test split to ensure balanced target variable distribution.

- **Pipeline:**
  - Indexed and one-hot encoded categorical features (`Area`, `VehBrand`, `VehGas`, `Region`).
  - Assembled and standardized numeric and encoded features.
  - Constructed a pipeline for model training and evaluation.

- **Model Training:**
  - **Poisson Regression:** Predicted `PP_ClaimNb`, evaluated with RMSE.
  - **Linear Regression (L1/L2):** Predicted `LogClaimNb`, evaluated with RMSE.
  - **Logistic Regression (L1/L2):** Predicted `NZClaim`, evaluated with accuracy.

- **Cross-Validation:**
  - Conducted with a 90/10 split on the training set.
  - Tested multiple `regParam` values, logging performance metrics.
  - Plotted validation curves to show the effect of regularization.

### Outputs

- **Model Coefficients:** Printed for each model.
- **Performance Metrics:** RMSE for regression models and accuracy for classification models.
- **Validation Curves:** Saved as PNG files in the `Output` directory.


## T3: Model Evaluation and Parameter Tuning for Collaborative Filtering

**Objective:** Enhance the performance of collaborative filtering models using movie and user score data through evaluation and parameter tuning, and explore user clusters using K-Means clustering.

### Methods Used

- **Model Evaluation:**
  - **Metrics:** Evaluated the performance of Alternating Least Squares (ALS) models using metrics such as RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and MSE (Mean Squared Error).
  - **Training Splits:** The dataset was split into different proportions (40%, 60%, 80%) to assess the model's performance under varying training data sizes.

- **Parameter Tuning:**
  - **ALS Settings:** Experimented with different ALS model settings, including default parameters and custom configurations (e.g., `rank=14`, `maxIter=17`), to identify the optimal hyperparameters.
  - **Validation Curves:** Generated validation curves to visualize and compare the performance of ALS models with different settings across various training data splits.

- **Clustering Analysis:**
  - **K-Means Clustering:** Applied K-Means clustering to user factors derived from the ALS model to identify user clusters.
  - **Cluster Evaluation:** Evaluated cluster sizes and identified the top 5 clusters for each training split.
  - **Genre Analysis:** For the largest user cluster in each split, the top 10 genres were extracted based on the users' highest-rated movies.

### Outputs

- **Model Performance Metrics:** Plotted the performance of the ALS models with different settings across various training data splits. The results are saved as `Model_Performance_Metrics.png`.
- **Top 5 Cluster Sizes:** Visualized the sizes of the top 5 user clusters identified by K-Means clustering for different training data splits. The results are saved as `Top_5_Cluster_Sizes.png`.
- **Top 10 Genres:** Displayed the top 10 genres for the largest user clusters across each training split, providing insights into users' preferences based on their ratings.


## T4: Log Analysis from NASA Access Logs

**Objective:** Analyze NASA web server access logs to extract and visualize insights on web traffic, focusing on requests from Germany, Canada, and Singapore.

## Data Extraction

- **Log Parsing:** Extracted `host`, `timestamp`, `request`, `HTTP_reply_code`, and `bytes_in_reply` from the raw log file.
- **Country-Specific Data:** Filtered logs to isolate requests from Germany (`.de`), Canada (`.ca`), and Singapore (`.sg`).

## Analysis

- **Request Counts:**
  - Total and unique requests from each country.
  - Top 9 most frequent hosts in each country.
  
- **Percentage Distribution:**
  - Calculated the percentage of requests made by the top 9 hosts compared to the total requests.
  - Visualized this distribution using pie charts.

- **Temporal Analysis:**
  - Created heatmaps to show the distribution of requests over time from the most frequent host in each country.

## Visualizations

- **Pie Charts:** Show percentage distribution of requests among top hosts and others.
- **Heatmaps:** Display visit patterns across days and hours for the most frequent hosts.

## Outputs

- **Charts:** Saved as `.jpg` files in the `Output` directory, showing the distribution and temporal analysis for each country.

## Code Structure


## Project Dependencies

- Apache Spark (version compatible with the scripts)
- Python packages: `pyspark`, `matplotlib`, `seaborn`, `pandas`

## Data

The data files required for Insurance Claims Prediction & Log Analysis scripts are:

- **Insurance Claims Prediction:** `../Data/freMTPL2freq.csv` available in `arfff` format
- **Log Analysis:** `../Data/NASA_access_log_Jul95.gz

## Note

This project was initially developed to run on the Sheffield Advanced Research Computer (ShaRC) cluster. Some analyses were performed on private data that is not accessible, which is why, even though the code is correct, it is unusable for others. The project primarily serves as a showcase of skills in working with large-scale datasets, distributed file systems (DFS), and Spark. Make sure to adapt paths and configurations if running in a different environment.
