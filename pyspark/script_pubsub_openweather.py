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
import math

project_number = 684093064430
location = "europe-central2" 
subscription_id1 = "weather-spark"
subscription_id2 = "stock-spark"

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

sdf = (spark.readStream.format("pubsublite")
    .option("pubsublite.subscription", f"projects/{project_number}/locations/{location}/subscriptions/{subscription_id1}",).load())

# Convert the binary "data" column to a string and all columns inside to string through schema
sdf_parsed = sdf.withColumn("data", F.from_json(F.col("data").cast(StringType()), weather_schema))

# Cast all columns to proper types
sdf_casted = sdf_parsed.select(
    F.col("data.temp").cast(DoubleType()).alias("temp"),
    F.col("data.weather_description").alias("weather_description"),
    F.col("data.visibility").cast(IntegerType()).alias("visibility"),
    F.col("data.pressure").cast(IntegerType()).alias("pressure"),
    F.col("data.clouds").cast(IntegerType()).alias("clouds"),
    F.col("data.feels_like").cast(DoubleType()).alias("feels_like"),
    F.col("data.temp_max").cast(DoubleType()).alias("temp_max"),
    F.col("data.weather_main").alias("weather_main"),
    F.col("data.temp_min").cast(DoubleType()).alias("temp_min"),
    F.col("data.humidity").cast(IntegerType()).alias("humidity"),
    F.col("data.wind_speed").cast(DoubleType()).alias("wind_speed"),
    F.col("data.timestamp").cast(DoubleType()).cast(TimestampType()).alias("timestamp"),
    F.col("publish_timestamp"),
)

## Transform columns into features
df_onehot = sdf_casted.select(F.col("timestamp"), F.col("weather_main"))
df_onehot = df_onehot.withColumn("date", F.to_date(df_onehot.timestamp))
df_onehot = df_onehot.withColumn("hour", F.hour(df_onehot.timestamp))
df_onehot = df_onehot.withColumn("day_of_week", F.dayofweek(df_onehot.timestamp)).withColumn(
    "month", F.month(df_onehot.timestamp)
) 
weather_main_values = [
    "wm_thunderstorm",
    "wm_drizzle",
    "wm_rain",
    "wm_snow",
    "wm_clear",
    "wm_clouds",
]
df_onehot = (
    df_onehot.withColumn(
        "wm_thunderstorm",
        F.when(df_onehot.weather_main == "Thunderstorm", 1).otherwise(0),
    )
    .withColumn(
        "wm_drizzle", F.when(df_onehot.weather_main == "Drizzle", 1).otherwise(0)
    )
    .withColumn("wm_rain", F.when(df_onehot.weather_main == "Rain", 1).otherwise(0))
    .withColumn("wm_snow", F.when(df_onehot.weather_main == "Snow", 1).otherwise(0))
    .withColumn("wm_clear", F.when(df_onehot.weather_main == "Clear", 1).otherwise(0))
    .withColumn("wm_clouds", F.when(df_onehot.weather_main == "Clouds", 1).otherwise(0))
    .withColumn("pom", reduce(add, [F.col(x) for x in weather_main_values]))
)

df_onehot = df_onehot.withColumn("wm_other", F.when(df_onehot.pom == 0, 1).otherwise(0))
df_onehot = df_onehot.drop(F.col("weather_main")).drop(F.col("pom"))

##
df_input = sdf_casted.join(df_onehot, ["timestamp"])

### Load models

model_tomtom=SparkXGBRegressorModel.load("/models/tomtom_model")
model_stock=SparkXGBRegressorModel.load("/models/stock_model")

### assemble features for prediction

assembler = VectorAssembler(
    inputCols=[x.name for x in df_input.schema if x.name not in ["date","timestamp","weather_main","weather_description","publish_timestamp"]],
    outputCol="features",
)

df_assembled=assembler.transform(df_input)

#predict TomTom and Stock

tomtom_results=model_tomtom.transform(df_assembled).withColumnRenamed("prediction", "tomtom_prediction")

combined_results=model_stock.transform(tomtom_results).withColumnRenamed("prediction", "stock_prediction")

### stock stream

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
stock_sdf_parsed = sdf.withColumn("data", F.from_json(F.col("data").cast(StringType()), stock_schema))

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
)

# combine combined_results (timestamp, publish_timestamp) with stock_sdf_casted (datetime, publish_timestamp?) (using timestamp??)
# https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#stream-stream-joins

# deduplication?
# https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#streaming-deduplication

# final columns
results = combined_results.select(F.col("date"), F.col("hour"), 
                                  F.col("temp"), F.col("pressure"), F.col("clouds"), F.col("clouds"), F.col("feels_like"), F.col("temp_max"), F.col("temp_min"), F.col("humidity"), F.col("wind_speed"), F.col("weather_main"), F.col("weather_description"), 
                                  # stock columns
                                    F.col("tomtom_prediction"), F.col("stock_prediction"))

# Define your output sink (e.g., write to console for testing)
#query = combined_results.writeStream.outputMode("append").format("console").start()

### WRITING TO BIGTABLE

client = bigtable.Client(project="bda-project-412623", admin=True)
instance = client.instance("bda-bigtable")
table = instance.table("stream")
timestamp = datetime.datetime.utcnow()

time_columns = ["date", "hour"], #time column group
weather_columns =["temp", "pressure", "clouds", "clouds", "feels_like", "temp_max", "temp_min", "humidity", "wind_speed", "weather_main", "weather_description"] #weather column group
# stock_columns =            # stock column group
predictions_columns = ["tomtom_prediction", "stock_prediction"] # predictions column group

def process_row(row):
    # Write row to storage
    row_key = "current"
    row = table.direct_row(row_key)
    for column in time_columns:
        row.set_cell("time", column, str(row.__getitem__(column)), timestamp)
    for column in weather_columns:
        row.set_cell("weather", column, str(row.__getitem__(column)), timestamp)
    # for column in stock_columns:
    #     row.set_cell("stock", column, str(row.__getitem__(column)), timestamp)
    for column in predictions_columns:
        row.set_cell("predictions", column, str(row.__getitem__(column)), timestamp)
    row.commit()


query = results.writeStream.outputMode("append").foreach(process_row).start()

# # Wait 120 seconds (must be >= 60 seconds) to start receiving messages.
query.awaitTermination(120)
query.stop()