import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as sql
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import split, regexp_extract
from pyspark.sql.types import DecimalType, IntegerType, DoubleType
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# Create the Spark session
spark = SparkSession \
    .builder.master("local[8]")\
    .config("spark.local.dir", "/fastdata/acq22vv") \
    .appName("Higgs_Particle_Classification") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

def log_confusion_matrix(predictions_rdd, model_name):
    metrics = MulticlassMetrics(predictions_rdd)
    confusion_matrix = metrics.confusionMatrix().toArray()

    with open(f"../Output/{model_name}_confusion_matrix.txt", "w") as f:
        f.write(f"Confusion Matrix for {model_name}:\n")
        f.write(str(confusion_matrix))

def log_metrics(predictions_rdd, model_name):
    metrics = MulticlassMetrics(predictions_rdd)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1 = metrics.fMeasure(1.0)

    with open(f"../Output/{model_name}_metrics.txt", "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")

def log_training_time(start_time, end_time, model_name):
    with open("../Output/training_times.txt", "a") as f:
        f.write(f"{model_name} Training Time: {end_time - start_time} seconds\n")

def log_feature_importances(model, model_name):
    feature_importances = model.featureImportances.toArray()
    feature_importances_str = " ".join(str(float(i)) for i in feature_importances)

    with open(f"../Output/{model_name}_feature_importances.txt", "w") as f:
        f.write(f"Feature Importances for {model_name}:\n")
        f.write(feature_importances_str)

def log_model_parameters(model, model_name):
    params = model.extractParamMap()
    with open("../Output/model_parameters.txt", "a") as f:
        f.write(f"{model_name} Parameters:\n")
        for param, value in params.items():
            f.write(f"{param.name}: {value}\n")

# Reading the CSV file and splitting the data to columns
Data = spark.read.csv("../Data/HIGGS.csv", sep=',')
col_name = Data.columns
for i in range(len(Data.columns)):
    Data = Data.withColumn(col_name[i], Data[col_name[i]].cast(DoubleType()))
Data = Data.withColumnRenamed("_c0", "label")

# Splitting the dataset
(Data_scaled, Data_remain) = Data.randomSplit([0.01, 0.99], 40)
(traindata, testdata) = Data_scaled.randomSplit([0.7, 0.3], 40)

# Random Forest
assembler = VectorAssembler(inputCols=col_name[1:len(col_name)], outputCol='features')
model_rf = RandomForestClassifier(labelCol="label", featuresCol="features")
rf_pipeline = Pipeline(stages=[assembler, model_rf])

crossval_rf = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=ParamGridBuilder()
        .addGrid(model_rf.numTrees, [20, 25, 30])
        .addGrid(model_rf.featureSubsetStrategy, ["log2", "auto", "all"])
        .addGrid(model_rf.subsamplingRate, [0.5, 0.75, 1.0])
        .build(),
    evaluator=MulticlassClassificationEvaluator(),
    numFolds=3,
    seed=40
)

start_time = time.perf_counter()
rf_modelcv = crossval_rf.fit(traindata)
end_time = time.perf_counter()

best_rf_model = rf_modelcv.bestModel.stages[1]
rf_numTrees, rf_featureSubsetStrategy, rf_subsamplingRate = (
    best_rf_model.getNumTrees,
    best_rf_model.getFeatureSubsetStrategy(),
    best_rf_model.getSubsamplingRate()
)

log_training_time(start_time, end_time, "RandomForest")
log_model_parameters(best_rf_model, "RandomForest")

rf_predict = rf_modelcv.transform(testdata).select("prediction", "label")
rf_prediction_rdd = rf_predict.rdd.map(lambda row: (float(row['prediction']), float(row['label'])))

rf_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC").evaluate(rf_predict)
accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(rf_predict)

print("\nThe accuracy for the random forest is:", accuracy)
print("\nThe area under the curve for the random forest is:", rf_auc)

log_confusion_matrix(rf_prediction_rdd, "RandomForest")
log_metrics(rf_prediction_rdd, "RandomForest")
log_feature_importances(best_rf_model, "RandomForest")

# Gradient Boosting
gbt_model = GBTClassifier(labelCol="label", featuresCol="features")
gbt_pipeline = Pipeline(stages=[assembler, gbt_model])
param_grid = (ParamGridBuilder()
              .addGrid(gbt_model.maxIter, [8, 11, 14])
              .addGrid(gbt_model.maxDepth, [5, 6, 7])
              .addGrid(gbt_model.subsamplingRate, [0.6, 0.8, 1.0])
              .build())

cv_gbt = CrossValidator(estimator=gbt_pipeline,
                        estimatorParamMaps=param_grid,
                        evaluator=MulticlassClassificationEvaluator(),
                        numFolds=3,
                        seed=40)

start_time = time.perf_counter()
gbt_modelcv = cv_gbt.fit(traindata)
end_time = time.perf_counter()

gbt_maxIter = gbt_modelcv.bestModel.stages[-1].getMaxIter()
gbt_MaxDepth = gbt_modelcv.bestModel.stages[-1].getMaxDepth()
gbt_SubsamplingRate = gbt_modelcv.bestModel.stages[-1].getSubsamplingRate()

log_training_time(start_time, end_time, "GBT")
log_model_parameters(gbt_modelcv.bestModel, "GBT")

gbt_predict = gbt_modelcv.transform(testdata).select("prediction", "label")
gbt_prediction_rdd = gbt_predict.rdd.map(lambda row: (float(row['prediction']), float(row['label'])))

auc_gbt = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC").evaluate(gbt_predict)
accuracy_gbt = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(gbt_predict)

print(f"\nGBT Accuracy: {accuracy_gbt}")
print(f"\nGBT AUC: {auc_gbt}")

log_confusion_matrix(gbt_prediction_rdd, "GBT")
log_metrics(gbt_prediction_rdd, "GBT")
log_feature_importances(gbt_modelcv.bestModel, "GBT")

# Use the full dataset
Data = assembler.transform(Data)
(train_f, test_f) = Data.randomSplit([0.7, 0.3], 40)

rf_full = RandomForestClassifier(labelCol="label", featuresCol="features",
                                numTrees=rf_numTrees,
                                featureSubsetStrategy=rf_featureSubsetStrategy,
                                subsamplingRate=rf_subsamplingRate)

gbt_full = GBTClassifier(labelCol="label", featuresCol="features",
                        maxIter=gbt_maxIter,
                        maxDepth=gbt_MaxDepth,
                        subsamplingRate=gbt_SubsamplingRate)

start_time = time.perf_counter()
rf_model_full = rf_full.fit(train_f)
end_time = time.perf_counter()
log_training_time(start_time, end_time, "RandomForest_Full")

start_time = time.perf_counter()
gbt_model_full = gbt_full.fit(train_f)
end_time = time.perf_counter()
log_training_time(start_time, end_time, "GBT_Full")

rf_prediction = rf_model_full.transform(test_f).select("prediction", "label")
rf_full_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC").evaluate(rf_prediction)
print("\nThe Random Forest full dataset AUC:", rf_full_auc)

gbt_prediction = gbt_model_full.transform(test_f).select("prediction", "label")
gbt_full_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC").evaluate(gbt_prediction)
print("\nThe GBT full dataset AUC value:", gbt_full_auc)
