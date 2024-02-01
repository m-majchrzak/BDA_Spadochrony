from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)

project_number = 684093064430
location = "europe-central2" 
subscription_id = "stock-spark"

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

stock_schema = StructType([
    StructField("volume", StringType(), True),
    StructField("vwap", StringType(), True),
    StructField("open", StringType(), True),
    StructField("close", StringType(), True),
    StructField("high", StringType(), True),
    StructField("low", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("transactions", StringType(), True),
    StructField("ticker", StringType(), True),
    StructField("status", StringType(), True),
    StructField("datetime", StringType(), True)
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
sdf_parsed = sdf.withColumn("data", from_json(col("data").cast(StringType()), stock_schema))


# Cast all columns to proper types
sdf_casted = sdf_parsed.select(
    col("data.volume").cast(IntegerType()).alias("volume"),
    col("data.vwap").cast(DoubleType()).alias("vmap"),
    col("data.open").cast(DoubleType()).alias("open"),
    col("data.close").cast(DoubleType()).alias("close"),
    col("data.high").cast(DoubleType()).alias("high"),
    col("data.low").cast(DoubleType()).alias("low"),
    col("data.transactions").cast(IntegerType()).alias("transactions"),
    col("data.ticker").cast(StringType()).alias("ticker"),
    col("data.status").cast(StringType()).alias("status"),
    col("data.datetime").cast(TimestampType()).alias("datetime"),
    col("publish_timestamp"),
)

# Define your output sink (e.g., write to console for testing)
query = sdf_casted.writeStream.outputMode("append").format("console").start()

# Wait 120 seconds (must be >= 60 seconds) to start receiving messages.
query.awaitTermination(120)
query.stop()