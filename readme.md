# Cluster-Based Data Analysis and Machine Learning

This project consists of several Spark-based analyses, including particle physics classification, web log analysis and collaborative filtering. The code is designed to run on the Sheffield Advanced Research Computer (ShaRC) cluster but can be adapted to other Spark environments.

## Project Structure

The project is divided into the following tasks:

### T1: Higgs Particle Classification

**Objective:** Classify Higgs boson particles using machine learning algorithms.

**Methods Used:**
- **Data Preprocessing:** Load and clean the Higgs boson dataset.
- **Model Training:** Train Random Forest and Gradient Boosting models.
- **Evaluation:** Assess model performance using metrics like ROC AUC.

### T2: Insurance Claims Prediction

**Objective:** Predict insurance claims using Generalized Linear Models (GLM) and other regression techniques.

**Methods Used:**
- **Data Preprocessing:** Handle categorical variables and scale numerical features.
- **Model Training:** Train GLM with Poisson family, Linear Regression, and Logistic Regression models.
- **Evaluation:** Evaluate models using RMSE and AUC.

### T3: Model Evaluation and Parameter Tuning

**Objective:** Evaluate and tune machine learning models for improved performance.

**Methods Used:**
- **Model Evaluation:** Evaluate models using metrics like RMSE and AUC.
- **Parameter Tuning:** Perform hyperparameter tuning using validation curves.

### T4: Log Analysis from NASA Access Logs

**Objective:** Analyze NASA access logs to identify request patterns and generate visualizations.

**Methods Used:**
- **Data Extraction:** Parse and extract fields from log files.
- **Request Analysis:** Count requests from different countries and identify top hosts.
- **Visualization:** Create pie charts and heatmaps to visualize request distributions.

## Project Dependencies

- Apache Spark (version compatible with the scripts)
- Python packages: `pyspark`, `matplotlib`, `seaborn`, `pandas`

## Data

The data files required for Insurance Claims Prediction & Log Analysis scripts are:

- **Insurance Claims Prediction:** `../Data/freMTPL2freq.csv` available in `arfff` format
- **Log Analysis:** `../Data/NASA_access_log_Jul95.gz

## Note

This project was initially developed to run on the Sheffield Advanced Research Computer (ShaRC) cluster. Some analyses were performed on private data that is not accessible, which is why, even though the code is correct, it is unusable for others. The project primarily serves as a showcase of skills in working with large-scale datasets, distributed file systems (DFS), and Spark. Make sure to adapt paths and configurations if running in a different environment.
