from pyspark.sql import SparkSession
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, asc, split as sparkSplit, \
                                    explode, desc, avg
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans

# Initialize Spark session
spark = SparkSession.builder \
    .master("local[6]") \
    .appName("ALS and Clustering Analysis") \
    .config("spark.local.dir", "/fastdata/acq22vv") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

print("\n-----------------------------")
print("Starting data processing and model evaluation...\n")

# Load and preprocess data
rawdata = spark.read.csv('./Data/ratings.csv', header=True)
moviesdata = spark.read.csv('./Data/movies.csv', header=True).cache()

# Convert data types to DoubleType
processed_data = rawdata
for column in rawdata.columns:
    processed_data = processed_data.withColumn(column, col(column).cast(DoubleType()))
processed_data.cache()

# Define seed and training splits
seed = 40
train_splits = [0.4, 0.6, 0.8]

def train_and_evaluate_als_model(als_model):
    """Train and evaluate ALS model for various training splits."""
    results = {}
    user_factors = {}
    training_data_collection = {}
    for split in train_splits:
        rw_count = int(processed_data.count() * split)
        training_data = processed_data.orderBy(asc("timestamp")).limit(rw_count).cache()
        test_data = processed_data.subtract(training_data)
        
        print(f"Split: {int(split*100)}% - Train count: {training_data.count()}, Test count: {test_data.count()}")
        
        model = als_model.fit(training_data)
        user_factors_split = model.userFactors.cache()
        predictions = model.transform(test_data)
        
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        evaluator.setMetricName("mse")
        mse = evaluator.evaluate(predictions)
        
        print(f"RMSE = {rmse} | MAE = {mae} | MSE = {mse}")
        
        results[split] = [rmse, mae, mse]
        user_factors[split] = user_factors_split
        training_data_collection[split] = training_data
    return results, user_factors, training_data_collection

# ALS model with default settings
print("\nEvaluating ALS model with default settings...")
als_default = ALS(userCol="userId", itemCol="movieId", seed=seed, coldStartStrategy="drop")
default_results, _, _ = train_and_evaluate_als_model(als_default)

# ALS model with custom settings
print("\nEvaluating ALS model with custom settings...")
als_custom = ALS(userCol="userId", itemCol="movieId", seed=seed, coldStartStrategy="drop", rank=14, maxIter=17)
custom_results, user_factors_custom, training_data_custom = train_and_evaluate_als_model(als_custom)

print("\nPlotting evaluation metrics...\n")

# Prepare data for plotting
x_labels = [f"{int(i*100)}%" for i in train_splits]
default_metrics = np.array(list(default_results.values()))
custom_metrics = np.array(list(custom_results.values()))

# Create plot
plt.figure()
metrics_labels = ["RMSE", "MAE", "MSE"]
for i, metric in enumerate(metrics_labels):
    plt.plot(x_labels, default_metrics[:, i], 'o--', alpha=0.4, label=f"Default - {metric}")
    plt.plot(x_labels, custom_metrics[:, i], 'o-', label=f"Custom - {metric}")

plt.title("Performance of ALS Models with Different Settings")
plt.xlabel("Training Data Splits")
plt.ylabel("Metric Score")
plt.legend()
plt.savefig("Output/Model_Performance_Metrics.png", bbox_inches='tight')

# Print evaluation metrics table
metrics_df = pd.DataFrame(default_metrics, columns=[f"Default {label}" for label in metrics_labels], 
                          index=x_labels)
metrics_df = pd.concat([metrics_df, pd.DataFrame(custom_metrics, columns=[f"Custom {label}" for label in metrics_labels], index=x_labels)], axis=1)
print(metrics_df)

print("\nEvaluating K-Means Clustering...\n")

# Initialize KMeans model
kmeans = KMeans(k=25, seed=seed)
clusters = {}
top_genres = {}

# Perform KMeans clustering for each training split
for split in train_splits:
    model = kmeans.fit(user_factors_custom.get(split))
    summary = model.summary
    
    top_clusters = sorted(summary.clusterSizes, reverse=True)[:5]
    clusters[split] = top_clusters
    
    transformed_data = model.transform(user_factors_custom.get(split))
    largest_cluster_id = transformed_data.groupBy('prediction').count() \
                                         .orderBy(desc("count")) \
                                         .first()['prediction']
    largest_cluster_users = transformed_data.filter(transformed_data.prediction == largest_cluster_id) \
                                             .select("id").distinct()
    
    top_movies = training_data_custom.get(split).join(largest_cluster_users, col("userId") == col("id"), "inner") \
                                          .groupBy("movieId") \
                                          .agg(avg("rating").alias("avg_rating")) \
                                          .filter(col("avg_rating") >= 4)
    
    user_movies = moviesdata.withColumnRenamed("movieId", "movieId2") \
                            .join(top_movies, col("movieId") == col("movieId2"), "inner") \
                            .drop("movieId2")
    
    top_genres_split = user_movies.withColumn("genres_each", sparkSplit("genres", "\|")) \
                                  .select(col("movieId"), col("title"), explode(col("genres_each")).alias("genre")) \
                                  .groupBy("genre") \
                                  .count() \
                                  .orderBy(desc("count")) \
                                  .limit(10) \
                                  .select("genre")
    
    top_genres[str(int(split*100)) + "%"] = [row['genre'] for row in top_genres_split.collect()]

# Plot cluster sizes
print("\nPlotting top 5 cluster sizes...\n")

plt.figure()
bar_width = 0.15
x_positions = np.arange(len(train_splits))

for i, (split, sizes) in enumerate(clusters.items()):
    for rank, size in enumerate(sizes):
        plt.bar(x_positions[i] + (rank - 2) * bar_width, size, bar_width, label=f'Rank {rank + 1}' if i == 0 else "")

plt.ylabel('Cluster Size')
plt.xlabel('Training Data Splits')
plt.title('Top 5 Cluster Sizes for Different Training Splits')
plt.xticks(x_positions, x_labels)
plt.legend()
plt.savefig("Output/Top_5_Cluster_Sizes.png", bbox_inches='tight')

# Print top 10 genres
print("\nTop 10 Genres from Largest Clusters for Each Training Split:\n")
print(pd.DataFrame(top_genres))

print("\nProcessing complete.")
