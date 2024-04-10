from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, StructField, FloatType, BooleanType, IntegerType
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
import sys

def get_value_schema():
    return StructType([StructField("timestep", IntegerType(), False),
                       StructField("vehicle_CO", FloatType(), False),
                       StructField("vehicle_CO2", FloatType(), False),
                       StructField("vehicle_HC", FloatType(), False),
                       StructField("vehicle_NOx", FloatType(), False),
                       StructField("vehicle_PMx", FloatType(), False),
                       StructField("vehicle_fuel", FloatType(), False),
                       StructField("vehicle_id", StringType(), False),
                       StructField("vehicle_lane", StringType(), False),
                       StructField("vehicle_noise", FloatType(), False),
                       StructField("vehicle_speed", FloatType(), False),
                       StructField("vehicle_type", StringType(), False),
                       StructField("vehicle_waiting", BooleanType(), False),
                       StructField("vehicle_x", FloatType(), False),
                       StructField("vehicle_y", FloatType(), False)])


# def make_predict_fun(kmeans):

#     #@pandas_udf(get_output_schema(), functionType=PandasUDFType.GROUPED_MAP)
#     def predict(df):
#         model = kmeans[df["time_period"]]
#         res = model.transform(df)
#         return res
    
#     return predict



def main(args):
    kafka_broker = "kafka:9092"
    source_topic = "traffic_data"
    sink_topic = "predictions"
    time_period_size_minutes = int(args[1])
    time_period_size_sim = time_period_size_minutes * 60
    time_period_num = int(args[2])
    hdfs = "hdfs://namenode:9000/user/root"

    spark = SparkSession.builder.appName("SmartCityMobility").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    kmeans_pipelines = dict()
    for i in range(0, time_period_num):
        model_path = f'/models/k_means/model_{i}'
        kmeans_pipelines[i] = PipelineModel.load(hdfs + model_path)

    lreg_pipeline = PipelineModel.load(hdfs + '/models/lreg')

    #kmeans = kmeans_pipeline.stages[-1]
    #centers = []
    #for i, cluster in enumerate(kmeans.clusterCenters()):
    #    centers.append((i, cluster[0].item(), cluster[1].item()))

    #centers_df = spark.createDataFrame(centers, schema=["cluster_id", "lat", "lng"])
    #centers_df.show()

    data = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("startingOffsets", "earliest") \
            .option("subscribe", source_topic) \
            .load()

    data = data.selectExpr("timestamp as event_time", "CAST(key AS STRING)", "CAST(value AS STRING)", "CAST(offset as int)")
    schema = get_value_schema()
    data_deserialized = data.withColumn("value", F.from_json(data["value"], schema=schema))
    flat_df = data_deserialized.selectExpr("event_time", "offset", "key as vehicle_id", "value.timestep as timestep", "value.vehicle_fuel as vehicle_fuel",
                                           "value.vehicle_noise as vehicle_noise", "value.vehicle_type as vehicle_type",
                                           "value.vehicle_x as vehicle_x", "value.vehicle_y as vehicle_y", "value.vehicle_speed as vehicle_speed")
    
    chk_loc = chk_loc = f'{hdfs}/chk/raw'
    flat_df.withColumn("value", F.encode(F.to_json(F.struct(F.col("*"))), "iso-8859-1")) \
                    .writeStream \
                    .queryName("raw_data_query") \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", kafka_broker) \
                    .option("topic", "raw_data") \
                    .option("checkpointLocation", chk_loc) \
                    .outputMode("append") \
                    .start()

    flat_df = flat_df.withColumn("time_period", (flat_df["timestep"] / time_period_size_sim).cast("int"))

    cols = ["event_time", "offset", "time_period", "vehicle_id", "vehicle_x", "vehicle_y", "cluster_prediction", "vehicle_fuel", "vehicle_noise", "vehicle_type", "vehicle_speed"]

    for i in range(0, time_period_num):     
        df_time_period = flat_df.filter(flat_df["time_period"] == i)
        
        cluster_predictions = kmeans_pipelines[i].transform(df_time_period)
        cluster_predictions = cluster_predictions.select(cols)
        emission_predictions = lreg_pipeline.transform(cluster_predictions)

        lreg_udf = F.udf(lambda x: x if x >= 0 else abs(x), FloatType())
        predictions = emission_predictions.withColumn("emission_prediction", lreg_udf(emission_predictions["emission_prediction"])).select(cols + ["emission_prediction"])

        chk_loc = f'{hdfs}/chk/tp{i}'
        predictions.withColumn("value", F.encode(F.to_json(F.struct(F.col("*"))), "iso-8859-1")) \
                    .writeStream \
                    .queryName(f'time_period_{i}') \
                    .format("kafka") \
                    .option("kafka.bootstrap.servers", kafka_broker) \
                    .option("topic", sink_topic) \
                    .option("checkpointLocation", chk_loc) \
                    .outputMode("append") \
                    .start()
        
        # output_path =  f'{hdfs}/results/tp_{i}'
        # predictions.writeStream \
        #             .queryName(f'time_period_{i}') \
        #             .format("csv") \
        #             .option("path", output_path) \
        #             .option("checkpointLocation", chk_loc) \
        #             .outputMode("append") \
        #             .start()
                    

    #cluster_predictions = flat_df.writeStream.foreach(make_predict_kmeans(kmeans_pipelines)).format("csv").option("path", "/results/kmeans.csv").start()
    # kmeans_pipeline = get_pipeline(flat_df["timestep"])

    spark.streams.awaitAnyTermination()
    spark.stop()



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Missing parameters...")
        exit(1)
    main(sys.argv)