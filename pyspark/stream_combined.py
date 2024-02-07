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

PROJECT="grand-harbor-413313"
INSTANCE="bda-bigtable"
TABLE="stream"
project_number = 518523499774
location = "europe-central2" 
subscription_id1 = "weather-spark"
subscription_id2 = "stock-spark"

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()


# WEATHER STREAM

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
    StructField("timestamp", StringType(), True)
])

# Read streaming data from Pub/Sub Lite

sdf = (spark.readStream.format("pubsublite")
    .option("pubsublite.subscription", f"projects/{project_number}/locations/{location}/subscriptions/{subscription_id1}").load())
# Convert the binary "data" column to a string and all columns inside to string through schema
sdf_parsed = sdf.withColumn("data", F.from_json(F.col("data").cast(StringType()), weather_schema))
# query = sdf_parsed.writeStream.outputMode("append").format("console").start()
# query.awaitTermination(180)
# query.stop()

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
    F.col("data.timestamp").cast(TimestampType()).alias("timestamp"),
    F.col("publish_timestamp"),
)
#drop nonunique rows
#sdf_casted=sdf_casted.withWatermark("timestamp", "10 minutes").dropDuplicates(["timestamp"])


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

# predict TomTom and Stock

tomtom_results=model_tomtom.transform(df_assembled).withColumnRenamed("prediction", "tomtom_prediction")

combined_results=model_stock.transform(tomtom_results).withColumnRenamed("prediction", "stock_prediction")

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
    F.col("publish_timestamp"))
#drop nonunique rows 




# watermarks
combined_results = combined_results.withColumn("timestamp_weather", combined_results.timestamp)
combined_results = combined_results.withColumn("date_weather", F.to_date(combined_results.timestamp_weather))
#combined_results=combined_results.dropDuplicates(["timestamp_weather"])
combined_results_with_watermark = combined_results.withWatermark("timestamp_weather", "1 minutes")
# query = combined_results_with_watermark.writeStream.outputMode("append").format("console").start()
# query.awaitTermination(120)
# query.stop()

stock_sdf_casted = stock_sdf_casted.withColumn("timestamp_stock", stock_sdf_casted.datetime)
stock_sdf_casted = stock_sdf_casted.withColumn("date_stock", F.to_date(stock_sdf_casted.timestamp_stock))
#stock_sdf_casted = stock_sdf_casted.dropDuplicates(["timestamp_stock"])
stock_with_watermark=stock_sdf_casted.withWatermark("timestamp_stock", "1 minutes")
# query = stock_with_watermark.writeStream.outputMode("append").format("console").start()
# query.awaitTermination(120)
# query.stop()


# combine combined_results (timestamp, publish_timestamp) with stock_sdf_casted (datetime, publish_timestamp?) (using timestamp??)

joined_results = combined_results_with_watermark.join(
  stock_with_watermark,
  expr("""
    date_weather = date_stock AND
    timestamp_weather >= timestamp_stock - interval 30 seconds AND
    timestamp_weather <= timestamp_stock + interval 30 seconds
    """),
  "fullOuter"                 # can be "inner", "leftOuter", "rightOuter", "fllOuter", "leftSemi"
)

# https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#stream-stream-joins

# deduplication?
# https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#streaming-deduplication


# final columns
results = joined_results.select(F.col("timestamp_weather"),F.col("timestamp_stock"), F.col("date"), F.col("hour"),
                                 F.col("temp"), F.col("pressure"), F.col("clouds"), F.col("clouds"), F.col("feels_like"), F.col("temp_max"), F.col("temp_min"), F.col("humidity"), F.col("wind_speed"), F.col("weather_main"), F.col("weather_description"), 
                                 F.col("volume"), F.col("vmap"), F.col("open"), F.col("close"), F.col("high"), F.col("low"), F.col("transactions"), F.col("ticker"), F.col("status"),
                                F.col("tomtom_prediction"), F.col("stock_prediction")
                                )

# query = results.writeStream.outputMode("append").format("console").start()
# query.awaitTermination()

# WRITING TO BIGTABLE

# Define the columns for each column group
columns_to_save={
    "time":("timestamp_weather", "timestamp_stock", "date", "hour"),
    "weather":("temp", "pressure", "clouds", "feels_like", "temp_max", "temp_min", "humidity", "wind_speed", "weather_main", "weather_description"),
    "stock":("volume", "vmap", "open", "close", "high", "low", "transactions", "ticker", "status"),
    "predictions":("tomtom_prediction", "stock_prediction"),
}


def process_batch(batch_df, batch_id):
    client = bigtable.Client(project=PROJECT, admin=True)
    table=client.instance(INSTANCE).table(TABLE)
    timestamp=datetime.datetime.utcnow()
    rows_to_mutate=[]
    for row in batch_df.collect():
        row_key = row["timestamp_weather"].strftime("%Y-%m-%d_%H-%M")
        print(f"Data for {row_key} created")
        new_row = table.direct_row(row_key)
        for column_family,columns in columns_to_save.items():
                for column in columns:
                    new_row.set_cell(
                        column_family_id=column_family,
                        column=column,
                        value=str(row[column]),
                        timestamp=timestamp,
                    )
        rows_to_mutate.append(new_row)
    table.mutate_rows(rows_to_mutate)
    print(f"Batch {batch_id} processed successfully")
    return

# process batches of data
query = (
    results.writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", "/tmp/checkpoint_streaming/")
    .outputMode("append")
    .start()
)

query.awaitTermination()