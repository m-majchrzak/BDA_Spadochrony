import os
from functools import reduce
from operator import add

import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    StringType,
    IntegerType,
    TimestampType,
)
from pyspark.sql.window import Window
from xgboost.spark import SparkXGBRegressor


def cast_columns_to_schema(dataframe, schema):
    for field in schema.fields:
        column_name = field.name
        column_type = field.dataType
        dataframe = dataframe.withColumn(
            column_name, dataframe[column_name].cast(column_type)
        )
    return dataframe


def read_parquets_to_df(folder, schema):
    unioned_df = None
    base_path = f"/{folder}"
    glob_pattern = "*"
    jvm = spark.sparkContext._jvm
    fs_root = jvm.java.net.URI.create(base_path)
    conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(fs_root, conf)
    path_glob = jvm.org.apache.hadoop.fs.Path(os.path.join(base_path, glob_pattern))
    status_list = fs.globStatus(path_glob)
    for status in status_list:
        raw_path = status.getPath().toUri().getRawPath()
        df = spark.read.parquet(raw_path)
        df = cast_columns_to_schema(df, schema)
        if not unioned_df:
            unioned_df = df
        else:
            unioned_df = unioned_df.unionByName(df, allowMissingColumns=True)
    return unioned_df


weather_schema = StructType(
    [
        StructField("temp", DoubleType(), True),
        StructField("weather_description", StringType(), True),
        StructField("visibility", IntegerType(), True),
        StructField("pressure", IntegerType(), True),
        StructField("clouds", IntegerType(), True),
        StructField("feels_like", DoubleType(), True),
        StructField("temp_max", DoubleType(), True),
        StructField("weather_main", StringType(), True),
        StructField("temp_min", DoubleType(), True),
        StructField("humidity", IntegerType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("timestamp", TimestampType(), True),
    ]
)

spark = SparkSession.builder.appName("read-app").master("yarn").config("spark.dynamicAllocation.enabled", "false").getOrCreate()


# df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather/live")

# df_live = read_parquets_to_df(folder="openweather/live", schema=weather_schema)
# df_live = df_live.drop('wind_deg')
# df_live.show()


df_hist = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
df_hist = df_hist.drop("UNNAMED_FIELD")
# df_hist.show()

# df = df_hist.unionByName(df_live)
df = df_hist


df = df.withColumn("hour", F.hour(df.timestamp))
df = df.withColumn("day_of_week", F.dayofweek(df.timestamp)).withColumn(
    "month", F.month(df.timestamp)
)
df = df.drop(df.weather_description)

### One-hot encoding ###

df_onehot = df.select(F.col("timestamp"), F.col("weather_main"))
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
df_onehot = df_onehot.drop(df_onehot.pom).drop(df_onehot.weather_main)

df = df.join(df_onehot, ["timestamp"])
df = df.drop(F.col("weather_main")).sort("timestamp", ascending=True)


### TARGET VARIABLE ###

stock_schema = StructType(
    [
        StructField("volume", IntegerType(), True),
        StructField("vwap", DoubleType(), True),
        StructField("open", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("timestamp", IntegerType(), True),
        StructField("transactions", IntegerType(), True),
        StructField("ticker", StringType(), True),
        StructField("status", StringType(), True),
        StructField("datetime", TimestampType(), True),
    ]
)

df_hist_stock = read_parquets_to_df(folder="stock/historical", schema=stock_schema)
df_hist_stock = df_hist_stock.drop("UNNAMED_FIELD")

# df_live_stock = read_parquets_to_df(folder="stock/live", schema=stock_schema)
# df_stock = df_hist_stock.unionByName(df_live_stock)
df_stock = df_hist_stock

df_stock = df_stock.withColumn(
    "timestamp_from_datetime", df_stock.datetime.cast(dataType=TimestampType())
)
df_stock = df_stock.withColumn(
    "timestamp", F.col("timestamp_from_datetime") - F.expr("INTERVAL 20 minutes")
)
df_stock = df_stock.withColumn("timestamp_unix", F.unix_timestamp("timestamp"))

window_spec = (
    Window().orderBy("timestamp_unix").rangeBetween(0, 3600)
)  # 1-hour rolling mean

df_stock_agg = df_stock.select(
    F.col("timestamp"), F.col("timestamp_unix"), F.col("transactions")
)

df_stock_agg = df_stock_agg.withColumn(
    "number_of_transactions", F.sum("transactions").over(window_spec)
)
df_stock_agg = df_stock_agg.select(F.col("number_of_transactions"), F.col("timestamp"))
df = df.join(df_stock_agg, ["timestamp"])

df.agg(
    F.min("timestamp").alias("min_timestamp"), F.max("timestamp").alias("max_timestamp")
).show(truncate=False)

print(f"number of rows:{df.count()}")

df = df.drop(F.col("timestamp"))
df.show(truncate=False)

df = df.na.drop("any")

# ### TRAIN/TEST SPLIT ###

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

label_name = "number_of_transactions"

# # get a column with feature colums combined
assembler = VectorAssembler(
    inputCols=[x.name for x in train_df.schema if x.name != label_name],
    outputCol="features",
)

train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# ## MODEL ###
params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "n_estimators": 20,
    "min_child_weight": 70,
}

spark_reg_estimator = SparkXGBRegressor(
    features_col="features", label_col=label_name, tree_method="hist", **params
)

# #train the model
model = spark_reg_estimator.fit(train_df)


# predict on test data
train_predict_df = model.transform(train_df)
predict_df = model.transform(test_df)
train_predict_df.show()
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=label_name)

print("###\n\n\n")
print(params)

train_df.agg(F.avg(label_name)).show()
test_df.agg(F.avg(label_name)).show()

print(
    f"The MAE for train {evaluator.evaluate(train_predict_df, {evaluator.metricName: 'mae'})}"
)
print(
    f"The MAE for test {evaluator.evaluate(predict_df, {evaluator.metricName: 'mae'})}"
)
print("\n\n\n###")


# save the model
model.save("/models/stock_model")
