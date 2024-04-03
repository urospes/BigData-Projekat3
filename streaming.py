from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, StructField, FloatType, BooleanType, IntegerType
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
import os

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
                       StructField("vehicle_type", StringType(), False),
                       StructField("vehicle_waiting", BooleanType(), False),
                       StructField("vehicle_x", FloatType(), False),
                       StructField("vehicle_y", FloatType(), False)])



def main(args):
    kafka_broker = "kafka:9092"
    source_topic = "traffic_data"

    spark = SparkSession.builder.appName("SmartCityMobility").getOrCreate()

    kmeans_pipeline = PipelineModel.load('/kmeans_model')
    kmeans = kmeans_pipeline.stages[-1]
    centers = []
    for i, cluster in enumerate(kmeans.clusterCenters()):
        centers.append((i, cluster[0].item(), cluster[1].item()))

    centers_df = spark.createDataFrame(centers, schema=["cluster_id", "lat", "lng"])
    centers_df.show()

    data = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("startingOffsets", "earliest") \
            .option("subscribe", source_topic) \
            .load()

    data = data.selectExpr("timestamp as event_time", "CAST(key AS STRING)", "CAST(value AS STRING)", "CAST(offset as int)")
    schema = get_value_schema()
    data_deserialized = data.withColumn("value", F.from_json(data["value"], schema=schema))
    flat_df = data_deserialized.selectExpr("event_time", "offset", "key as vehicle_id",
                                           "value.vehicle_CO as vehicle_CO", "value.vehicle_CO2 as vehicle_CO2", "value.vehicle_HC as vehicle_HC",
                                           "value.vehicle_NOx as vehicle_NOx", "value.vehicle_PMx as vehicle_PMx", "value.vehicle_fuel as vehicle_fuel",
                                           "value.vehicle_noise as vehicle_noise", "value.vehicle_type as vehicle_type", "value.vehicle_x as vehicle_x",
                                           "value.vehicle_y as vehicle_y")
    
    emission_cols = ("vehicle_CO", "vehicle_CO2", "vehicle_HC", "vehicle_NOx", "vehicle_PMx")
    flat_df = flat_df.withColumn("emission_sum", sum(flat_df[col] for col in emission_cols))

    cluster_predictions = kmeans_pipeline.transform(flat_df)
    cluster_predictions = cluster_predictions.selectExpr("vehicle_id", "prediction")

    cluster_predictions.writeStream.format("console").outputMode("append").start()
    

    spark.streams.awaitAnyTermination()
    spark.stop()



if __name__ == "__main__":
    main(None)