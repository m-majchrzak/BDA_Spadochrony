from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)

project_number = 684093064430
location = "europe-central2" 
subscription_id = "weather-spark"

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

weather_schema = StructType([
    StructField("temp", StringType(), True),
    StructField("weather_description", StringType(), True),
    StructField("visibility", StringType(), True),
    StructField("pressure", StringType(), True),
    StructField("clouds", StringType(), True),
    StructField("feels_like", StringType(), True),
    StructField("temp_max", StringType(), True),
    StructField("weather_main", StringType(), True),
    StructField("temp_min", StringType(), True),
    StructField("humidity", StringType(), True),
    StructField("wind_speed", StringType(), True),
    StructField("timestamp", StringType(), True),
])

# Read streaming data from Pub/Sub Lite

sdf = (
    spark.readStream.format("pubsublite")
    .option(
        "pubsublite.subscription",
        f"projects/{project_number}/locations/{location}/subscriptions/{subscription_id}",
    )
    .load()
)

# Convert the binary "data" column to a string and all columns inside to string through schema
sdf_parsed = sdf.withColumn("data", from_json(col("data").cast(StringType()), weather_schema))

# Cast all columns to proper types
sdf_casted = sdf_parsed.select(
    col("data.temp").cast(DoubleType()).alias("temp"),
    col("data.weather_description").alias("weather_description"),
    col("data.visibility").cast(IntegerType()).alias("visibility"),
    col("data.pressure").cast(IntegerType()).alias("pressure"),
    col("data.clouds").cast(IntegerType()).alias("clouds"),
    col("data.feels_like").cast(DoubleType()).alias("feels_like"),
    col("data.temp_max").cast(DoubleType()).alias("temp_max"),
    col("data.weather_main").alias("weather_main"),
    col("data.temp_min").cast(DoubleType()).alias("temp_min"),
    col("data.humidity").cast(IntegerType()).alias("humidity"),
    col("data.wind_speed").cast(DoubleType()).alias("wind_speed"),
    col("data.timestamp").cast(DoubleType()).cast(TimestampType()).alias("timestamp"),
    col("publish_timestamp"),
)

# Define your output sink (e.g., write to console for testing)
query = sdf_casted.writeStream.outputMode("append").format("console").start()

# # Wait 120 seconds (must be >= 60 seconds) to start receiving messages.
query.awaitTermination(120)
query.stop()