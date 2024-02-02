from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType, TimestampType)
from xgboost.spark import SparkXGBRegressorModel
from pyspark.ml.feature import VectorAssembler
from functools import reduce
from operator import add
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
df_input = sdf_casted.join(df_onehot, ["timestamp"]).drop(F.col("weather_main"))

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

#print(tomtom_results)

combined_results=model_stock.transform(tomtom_results).withColumnRenamed("prediction", "stock_prediction")

#print(tomtom_results)

# Define your output sink (e.g., write to console for testing)
query = combined_results.writeStream.outputMode("append").format("console").start()

# # Wait 120 seconds (must be >= 60 seconds) to start receiving messages.
query.awaitTermination(120)
query.stop()