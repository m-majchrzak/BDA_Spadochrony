import os
from functools import reduce
from operator import add

import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
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

spark = (
    SparkSession.builder.appName("read-app")
    .master("yarn")
    .config("spark.dynamicAllocation.enabled", "false")
    .getOrCreate()
)


# df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather/live")

df_live = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
df_live = df_live.drop("wind_deg").drop("UNNAMED_FIELD")
# df_live.show()

# df_hist = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
# df_hist = df_hist.drop('UNNAMED_FIELD')
# df_hist.show()

# df = df_hist.unionByName(df_live)
# df = df_hist
df = df_live


df = df.withColumn("date", F.to_date(df.timestamp))
df = df.withColumn("hour", F.hour(df.timestamp))
df = df.withColumn("day_of_week", F.dayofweek(df.timestamp)).withColumn(
    "month", F.month(df.timestamp)
)  # .withColumn('min_since_midnight', hour(df.timestamp)*24+minute(df.timestamp)) \
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

# df.show()
### TARGET VARIABLE ###
tomtom_schema = StructType(
    [
        StructField("iconCategory", StringType(), True),  # Categorical column
        StructField("id", StringType(), True),  # Categorical column
        StructField(
            "probabilityOfOccurrence", StringType(), True
        ),  # Categorical column
        StructField("delay", DoubleType(), True),  # Numerical column
        StructField("magnitudeOfDelay", StringType(), True),  # Categorical column
        StructField("length", DoubleType(), True),  # Numerical column
        StructField(
            "observationTime", TimestampType(), True
        ),  # Categorical column (assuming it's a string)
    ]
)

# df_live_tomtom = read_parquets_to_df(folder="tomtom/live", schema=tomtom_schema)
# df_live_tomtom = df_live_tomtom.drop('timeValidity')

df_hist_tomtom = read_parquets_to_df(folder="tomtom/historical", schema=tomtom_schema)
df_hist_tomtom = df_hist_tomtom.drop("UNNAMED_FIELD")
df_hist_tomtom = df_hist_tomtom.drop("timeValidity")

# df = df_hist.unionByName(df_live)
df_tomtom = df_hist_tomtom
df_tomtom = (
    df_tomtom.withColumn(
        "timestamp", F.from_utc_timestamp(F.col("observationTime"), "Europe/Warsaw")
    )
    .withColumn("unix_timestamp", F.unix_timestamp(F.col("observationTime")))
    .filter(df_tomtom.iconCategory == 6)
)


df_tomtom_agg = (
    df_tomtom.filter(F.col("length").isNotNull())
    .groupBy(["timestamp", "unix_timestamp", "observationTime"])
    .agg(F.sum("length").alias("length_of_traffic_jams"))
)

window_spec = (
    Window().orderBy("unix_timestamp").rangeBetween(0, 3600)
)  # 1-hour rolling mean

df_tomtom_agg = df_tomtom_agg.withColumn(
    "avg_length_of_traffic_jams", F.avg("length_of_traffic_jams").over(window_spec)
)

df = df.join(df_tomtom_agg, ["timestamp"])
df.agg(
    F.min("observationTime").alias("min_observationTime"),
    F.max("observationTime").alias("max_observationTime"),
).show()
print(f"number of rows:{df.count()}")
df = (
    df.drop("length_of_traffic_jams")
    .drop("observationTime")
    .drop("unix_timestamp")
    .drop("timestamp")
    .drop("date")
)
df.show()


### TRAIN/TEST SPLIT ###

train_df, test_df = df.randomSplit([0.7, 0.3], seed=222)

label_name = "avg_length_of_traffic_jams"

# get a column with feature colums combined
assembler = VectorAssembler(
    inputCols=[x.name for x in train_df.schema if x.name != label_name],
    outputCol="features",
)
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)


print("train")
train_df.agg(F.avg(label_name)).show()
print("test")
test_df.agg(F.avg(label_name)).show()


params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "n_estimators": 15,
    "min_child_weight": 70,
}
## MODEL ###
spark_reg_estimator = SparkXGBRegressor(
    features_col="features", label_col=label_name, tree_method="hist", **params
)

# train the model
model = spark_reg_estimator.fit(train_df)


# predict on test data
train_predict_df = model.transform(train_df)
predict_df = model.transform(test_df)
train_predict_df.show()
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=label_name)

print("###")
print(params)
train_df.agg(F.avg(label_name)).show()
test_df.agg(F.avg(label_name)).show()

print(
    f"The MAE for train {evaluator.evaluate(train_predict_df, {evaluator.metricName: 'mae'})}"
)
print(
    f"The MAE for test {evaluator.evaluate(predict_df, {evaluator.metricName: 'mae'})}"
)
print("###")


# save the model
model.save("/models/tomtom_model")
