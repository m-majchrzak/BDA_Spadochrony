from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)
from xgboost.spark import SparkXGBRegressorModel
from pyspark.ml.feature import VectorAssembler
from functools import reduce
from operator import add
from google.cloud import bigtable
import datetime
from pyspark.sql.functions import udf
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import expr
from pyspark.sql.functions import coalesce

project_number = 684093064430
location = "europe-central2" 
subscription_id1 = "weather-spark"
subscription_id2 = "stock-spark"

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

# STOCK STREAM

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
stock_sdf = (spark.readStream.format("pubsublite")
    .option("pubsublite.subscription",f"projects/{project_number}/locations/{location}/subscriptions/{subscription_id2}",).load())
# Convert the binary "data" column to a string and all columns inside to string through schema
stock_sdf_parsed = stock_sdf.withColumn("data", F.from_json(F.col("data").cast(StringType()), stock_schema))

# Cast all columns to proper types
stock_sdf_casted = stock_sdf_parsed.select(
    F.col("data.volume").cast(IntegerType()).alias("volume"),
    F.col("data.vwap").cast(DoubleType()).alias("vmap"),
    F.col("data.open").cast(DoubleType()).alias("open"),
    F.col("data.close").cast(DoubleType()).alias("close"),
    F.col("data.high").cast(DoubleType()).alias("high"),
    F.col("data.low").cast(DoubleType()).alias("low"),
    F.col("data.transactions").cast(IntegerType()).alias("transactions"),
    F.col("data.ticker").cast(StringType()).alias("ticker"),
    F.col("data.status").cast(StringType()).alias("status"),
    F.col("data.datetime").cast(TimestampType()).alias("datetime"),
    F.col("publish_timestamp"),
)#.dropDuplicates(["datetime"])
#drop nonunique rows 

# if datetime is null (on weekends) fill it with publish_timestamp
# stock_sdf_casted = stock_sdf_casted.withColumn("datetime",coalesce(stock_sdf_casted.datetime,stock_sdf_casted.publish_timestamp))
# stock_sdf_casted = stock_sdf_casted.withColumn("date_stock", F.to_date(stock_sdf_casted.datetime))

# watermarks
#stock_with_watermark=stock_sdf_casted.withWatermark("datetime", "5 minutes")

# output to console
query = stock_sdf_casted.writeStream.outputMode("append").format("console").start()
query.awaitTermination(120)
query.stop()
