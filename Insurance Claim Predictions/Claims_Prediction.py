from pyspark.sql import SparkSession
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, log, count

# Spark logistics
spark = SparkSession.builder \
    .master("local[6]") \
    .appName("Claims Prediction") \
    .config("spark.local.dir", "/fastdata/acq22vv") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Load data
rawdata = spark.read.csv('./Data/freMTPL2freq.csv', header=True).cache()

# Exclude ID column
data = rawdata.select(rawdata.columns[1:])

# Convert numeric columns to double data type
num_cols = ['ClaimNb', 'Exposure', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
for column in num_cols:
    data = data.withColumn(column, data[column].cast(DoubleType()))

# Create NZClaim and LogClaimNb
data = data.withColumn("NZClaim", when(data["ClaimNb"] > 0, 1.0).otherwise(0.0))
data = data.withColumn("PP_ClaimNb", when(data["ClaimNb"] == 0, 0.5).otherwise(data["ClaimNb"]))
data = data.withColumn("PP_ClaimNb", col("PP_ClaimNb").cast("double"))
data = data.withColumn("LogClaimNb", log(col("PP_ClaimNb"))).drop(col("ClaimNb"))

# Perform a training and test split
seed = 14

def stratified_split(train_ratio, source_data):
    trainingData = spark.createDataFrame([], source_data.schema)
    testData = spark.createDataFrame([], source_data.schema)
    for row in source_data.groupBy("PP_ClaimNb").count() \
                         .withColumn("sampl_frac", col("count") / source_data.count()).collect():
        train = row["sampl_frac"] * train_ratio
        tempTrainingData, tempTestData = source_data \
            .filter(col("PP_ClaimNb") == row["PP_ClaimNb"]) \
            .randomSplit([train, (row["sampl_frac"] - train)], seed)
        trainingData = trainingData.unionAll(tempTrainingData)
        testData = testData.unionAll(tempTestData)
    return trainingData, testData

# Perform split of 70/30
trainingData, testData = stratified_split(0.7, data)
trainingData.cache()
testData.cache()

# Print out some stats
print("Total Data count:", data.count())
print("\nTrain split data count:", trainingData.count())
print("Test split data count:", testData.count())
print("Train + Test count:", trainingData.count() + testData.count())

print("\nDistribution of the data wrt to the Claims classes in full data...")
data.groupBy("PP_ClaimNb").agg(count("*").alias("count")) \
    .withColumn("percentage", col("count") / data.count() * 100).show()

print("\nDistribution of the data in train split...")
trainingData.groupBy("PP_ClaimNb").agg(count("*").alias("count")) \
    .withColumn("percentage", col("count") / trainingData.count() * 100).show()

print("\nDistribution of the data in test split...")
testData.groupBy("PP_ClaimNb").agg(count("*").alias("count")) \
    .withColumn("percentage", col("count") / testData.count() * 100).show()

# Create the stages for the ML pipeline
org_str_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
idxd_cols = ['Area_idx', 'VehBrand_idx', 'VehGas_idx', 'Region_idx']
str_indexer = StringIndexer(inputCols=org_str_cols, outputCols=idxd_cols)
ohe_cols = ['Area_ohe', 'VehBrand_ohe', 'VehGas_ohe', 'Region_ohe']
ohe = OneHotEncoder(inputCols=idxd_cols, outputCols=ohe_cols)
num_cols.remove('ClaimNb')
num_assembler = VectorAssembler(inputCols=num_cols, outputCol="Num_Features")
std_scaler = StandardScaler(inputCol="Num_Features", outputCol="Scld_Num_Features")
fv_assembler = VectorAssembler(inputCols=["Scld_Num_Features"] + ohe_cols, outputCol="features")

def train_predict_report(train_data, test_data, ml_model, label, variation, eval_typ):
    print("\n*** Performing training for", variation, "...")
    ml_stages = [str_indexer, ohe, num_assembler, std_scaler, fv_assembler, ml_model]
    pipeline = Pipeline(stages=ml_stages)
    ml_pipelineModel = pipeline.fit(train_data)
    predictions_test = ml_pipelineModel.transform(test_data)
    if eval_typ == "RMSE":
        evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
    elif eval_typ == "ACCURACY":
        evaluator = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction", metricName="accuracy")
    result_test = evaluator.evaluate(predictions_test)
    print(eval_typ, ":", result_test)
    print('Model Coefficients:')
    print(ml_pipelineModel.stages[-1].coefficients)
    predictions_train = ml_pipelineModel.transform(train_data)
    result_train = evaluator.evaluate(predictions_train)
    return (result_train, result_test)

def get_ml_model(model_type, regParam=0.001):
    if model_type == "glm_poisson":
        return GeneralizedLinearRegression(featuresCol='features', labelCol='PP_ClaimNb', maxIter=50, regParam=regParam, family='poisson', link='log')
    elif model_type == "linear_l1":
        return LinearRegression(featuresCol='features', labelCol='LogClaimNb', maxIter=50, regParam=regParam, elasticNetParam=1)
    elif model_type == "linear_l2":
        return LinearRegression(featuresCol='features', labelCol='LogClaimNb', maxIter=50, regParam=regParam, elasticNetParam=0)
    elif model_type == "logistic_l1":
        return LogisticRegression(featuresCol='features', labelCol='NZClaim', maxIter=50, regParam=regParam, elasticNetParam=1)
    elif model_type == "logistic_l2":
        return LogisticRegression(featuresCol='features', labelCol='NZClaim', maxIter=50, regParam=regParam, elasticNetParam=0)
    else:
        return None

# Run the training and evaluation for various models
train_predict_report(trainingData, testData, get_ml_model("glm_poisson", 0.001), 'PP_ClaimNb', "No. of claims (PP_ClaimNb) using Poisson Regression", eval_typ="RMSE")
train_predict_report(trainingData, testData, get_ml_model("linear_l1", 0.001), 'LogClaimNb', "LogClaimNb using Linear Regression and L1 Regularisation", eval_typ="RMSE")
train_predict_report(trainingData, testData, get_ml_model("linear_l2", 0.001), 'LogClaimNb', "LogClaimNb using Linear Regression and L2 Regularisation", eval_typ="RMSE")
train_predict_report(trainingData, testData, get_ml_model("logistic_l1", 0.001), 'NZClaim', "NZClaim using Logistic Regression and L1 Regularisation", eval_typ="ACCURACY")
train_predict_report(trainingData, testData, get_ml_model("logistic_l2", 0.001), 'NZClaim', "NZClaim using Logistic Regression and L2 Regularisation", eval_typ="ACCURACY")

# Perform cross-validation
cv_trainingData, cv_testData = stratified_split(0.9, trainingData)
cv_trainingData.cache()
cv_testData.cache()

print("Total Data count (for CV):", trainingData.count())
print("\nTrain split data count:", cv_trainingData.count())
print("Test split data count:", cv_testData.count())
print("Train + Test count:", cv_trainingData.count() + cv_testData.count())

reg_param_opts = [0.001, 0.01, 0.1, 1, 10]
print("Regularisation Parameter Options for cross-validation:", reg_param_opts)

vc_results = {}
for reg_opt in reg_param_opts:
    results = []
    results.append(train_predict_report(cv_trainingData, cv_testData, get_ml_model("glm_poisson", reg_opt), 'PP_ClaimNb', "PP_ClaimNb~Poisson Regression~regParam="+str(reg_opt), eval_typ="RMSE"))
    results.append(train_predict_report(cv_trainingData, cv_testData, get_ml_model("linear_l1", reg_opt), 'LogClaimNb', "LogClaimNb~Linear Regression(L1)~regParam="+str(reg_opt), eval_typ="RMSE"))
    results.append(train_predict_report(cv_trainingData, cv_testData, get_ml_model("linear_l2", reg_opt), 'LogClaimNb', "LogClaimNb~Linear Regression(L2)~regParam="+str(reg_opt), eval_typ="RMSE"))
    results.append(train_predict_report(cv_trainingData, cv_testData, get_ml_model("logistic_l1", reg_opt), 'NZClaim', "NZClaim~Logistic Regression(L1)~regParam="+str(reg_opt), eval_typ="ACCURACY"))
    results.append(train_predict_report(cv_trainingData, cv_testData, get_ml_model("logistic_l2", reg_opt), 'NZClaim', "NZClaim~Logistic Regression(L2)~regParam="+str(reg_opt), eval_typ="ACCURACY"))
    vc_results[reg_opt] = results

# Plot the model performance for each regParam tested
variations = ["PP_ClaimNb~Poisson Regression", "LogClaimNb~Linear Regression(L1)", "LogClaimNb~Linear Regression(L2)", "NZClaim~Logistic Regression(L1)", "NZClaim~Logistic Regression(L2)"]
xs = [str(i) for i in reg_param_opts]

for i, variation in enumerate(variations):
    train_s = []
    val_s = []
    for reg_opt, results in vc_results.items():
        val_s.append(results[i][1])
        train_s.append(results[i][0])
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.plot(xs, train_s, label="Training Score", color="orange")
    plt.plot(xs, val_s, label="Cross-validation score", color="blue")
    ax = plt.gca()
    ax.legend()
    plt.ylim(round(min(train_s + val_s) - min(train_s + val_s) * 0.001, 6), round(max(train_s + val_s) + max(train_s + val_s) * 0.001, 6))
    plt.title("Validation Curve for " + variation)
    plt.xlabel("regParam")
    plt.ylabel("Score (RMSE/Accuracy)")
    plt.savefig("Output/"+variation+".png", bbox_inches='tight')
