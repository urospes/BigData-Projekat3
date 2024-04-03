from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml.clustering import KMeans

def main(args):

    input_path = "./input/emission_data.csv"
    time_period_size_minutes = 60
    time_period_size_sim = time_period_size_minutes * 60

    spark = SparkSession.builder.appName("SmartCityMobility") \
                                .getOrCreate()

    dataset = spark.read.option("header", True) \
                        .option("inferSchema", True) \
                        .csv(input_path, sep=";")
    
    dataset.show(5)

    emission_cols = ("vehicle_CO", "vehicle_CO2", "vehicle_HC", "vehicle_NOx", "vehicle_PMx")
    dataset = dataset.withColumn("emission_sum", sum(dataset[col] for col in emission_cols)) \
                     .withColumn("time_period", (dataset["timestep_time"] / time_period_size_sim).cast("int"))
    
    time_periods = dataset.select("time_period").distinct().collect()

    for time_period in time_periods:
        time_period = time_period["time_period"]
        df = dataset.filter(dataset["time_period"] == time_period)

        # train_test split
        train, test = df.randomSplit(weights=[0.8, 0.2], seed=42)

        # TASK 1 ----------- K-MEANS CLUSTERING ----------------------

        # data preprocessing
        assembler = VectorAssembler(inputCols=("vehicle_x", "vehicle_y"), outputCol="features")

        # k-means
        kmeans = KMeans(seed=42, initSteps=3)
        pipeline = Pipeline(stages=[assembler, kmeans])

        #hyperparameter tuning with cross-validation
        paramGrid = ParamGridBuilder() \
                .addGrid(kmeans.k, [4, 5, 6, 7, 8, 9, 10, 11]) \
                .build()
        
        evaluator = ClusteringEvaluator(distanceMeasure="squaredEuclidean", metricName="silhouette")

        cv = CrossValidator(estimator=pipeline, evaluator=evaluator, numFolds=3, estimatorParamMaps=paramGrid)
        print("Clustering...")
        model = cv.fit(train)
        print(f'Clustering avg. Silhouette score: {model.avgMetrics}')
        best_model = model.bestModel

        # model evaluation
        predictions = best_model.transform(test)
        test_silhouette = evaluator.evaluate(predictions)
        print(f'Test Silhouette: {test_silhouette}')
        predictions[["features", "prediction"]].show(10, truncate=False)

        kmeans_path = f'k_means/model_{time_period}'
        best_model.write().overwrite().save(kmeans_path)



        # TASK 2 ----------- LINEAR REGRESSION  ----------------------

        # data preprocessing
        indexer = StringIndexer(inputCol="vehicle_type", outputCol="vehicle_type_indexed")
        oh_encoder = OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="vehicle_type_onehot")

        regression_cols = ("vehicle_fuel", "vehicle_noise", "vehicle_speed", "vehicle_type_onehot")
        target_col = "emission_sum"

        assembler = VectorAssembler(inputCols=regression_cols, outputCol="features")
        scaler = StandardScaler(inputCol=assembler.getOutputCol(), outputCol="features_scaled")

        # linear regression
        lreg = LinearRegression(featuresCol="features_scaled", labelCol=target_col, standardization=False)
        pipeline = Pipeline(stages=[indexer, oh_encoder, assembler, scaler, lreg])

        # hyperparameter tuning with cross-validation
        paramGrid = ParamGridBuilder() \
                    .addGrid(lreg.regParam, [0.1, 0.01, 0.001, 0.0001]) \
                    .addGrid(lreg.maxIter, [20, 50]) \
                    .build()
        
        evaluator = RegressionEvaluator(labelCol=target_col, metricName="rmse")

        cv = CrossValidator(estimator=pipeline, evaluator=evaluator, numFolds=5, estimatorParamMaps=paramGrid)
        print("Linear Regression...")
        model = cv.fit(train)
        print(f'Linear regression average RMSE: {model.avgMetrics}')
        best_model = model.bestModel

        # model evaluation
        predictions = best_model.transform(test)
        test_rmse = evaluator.evaluate(predictions)
        print(f'Test RMSE: {test_rmse}')
        predictions[["features", target_col, "prediction"]].show(10, truncate=False)

        lreg_path = f'./lreg/model_{time_period}'
        best_model.write().overwrite().save(lreg_path)



    spark.stop()


if __name__ == "__main__":
    main(None)