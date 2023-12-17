from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
import os
from operator import add
from functools import reduce
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

def mode_result(df, colname):
    count = df.groupBy(['date', 'hour', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour')
            .agg(max(struct(col('count'),
                                col(colname))).alias('max'))
            .select(col('date'), col('hour'), col(f'max.{colname}').alias(f'{colname}'))
            )
    return result

def mode_result_by_category_and_id(df, colname):
    count = df.groupBy(['date', 'hour', 'iconCategory', 'id', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour', 'iconCategory', 'id')
            .agg(max(struct(col('count'),
                                col(colname))).alias('max'))
            .select(col('date'), col('hour'), col('iconCategory'), col('id'), col(f'max.{colname}').alias(f'mode_{colname}'))
            )
    return result

def mode_result_by_category(df, colname):
    count = df.groupBy(['date', 'hour', 'iconCategory', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour', 'iconCategory')
            .agg(max(struct(col('count'),
                                col(colname))).alias('max'))
            .select(col('date'), col('hour'), col('iconCategory'), col(f'max.{colname}').alias(f'{colname}'))
            )
    return result

def cast_columns_to_schema(dataframe, schema):
    for field in schema.fields:
        column_name = field.name
        column_type = field.dataType
        dataframe = dataframe.withColumn(column_name, dataframe[column_name].cast(column_type))
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

weather_schema = StructType([
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
    ])

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()
 

#df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather/live")

df_live = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
df_live = df_live.drop('wind_deg')
#df_live.show()

#df_hist = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
#df_hist = df_hist.drop('UNNAMED_FIELD')
#df_hist.show()

#df = df_hist.unionByName(df_live)
#df = df_hist
df = df_live


df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp))
df = df.withColumn('day_of_week', dayofweek(df.timestamp)) \
    .withColumn('month', month(df.timestamp)) \
    #.withColumn('min_since_midnight', hour(df.timestamp)*24+minute(df.timestamp)) \
df = df.drop(df.weather_description)

# Aggregate by hour ###

df_agg = df.groupBy("date", "hour", "day_of_week", "month") \
    .agg(round(avg("temp"),2).alias("temp"), \
         round(avg("visibility"),2).alias("visibility"), \
         round(avg("pressure"),2).alias("pressure"), \
         round(avg("clouds"),2).alias("clouds"), \
         round(avg("feels_like"),2).alias("feels_like"), \
         round(avg("temp_max"),2).alias("temp_max"), \
         round(avg("temp_min"),2).alias("temp_min"), \
         round(avg("humidity"),2).alias("humidity"), \
         round(avg("wind_speed"),2).alias("wind_speed")) 
         

weather_main_result = mode_result(df, 'weather_main')
df_agg = df_agg.join(weather_main_result, ['date', 'hour']).sort("date", "hour", ascending=[True, True])
#df_agg.show()

### One-hot encoding ###
df_onehot = df_agg.select(col('date'), col('hour'), col('weather_main')) 
weather_main_values = ['wm_thunderstorm',
                       'wm_drizzle',
                       'wm_rain',
                       'wm_snow',
                       'wm_clear', 
                       'wm_clouds']
df_onehot = df_onehot.withColumn('wm_thunderstorm', when(df_onehot.weather_main == 'Thunderstorm', 1).otherwise(0)) \
    .withColumn('wm_drizzle', when(df_onehot.weather_main == 'Drizzle', 1).otherwise(0)) \
    .withColumn('wm_rain', when(df_onehot.weather_main == 'Rain', 1).otherwise(0)) \
    .withColumn('wm_snow', when(df_onehot.weather_main == 'Snow', 1).otherwise(0)) \
    .withColumn('wm_clear', when(df_onehot.weather_main == 'Clear', 1).otherwise(0)) \
    .withColumn('wm_clouds', when(df_onehot.weather_main == 'Clouds', 1).otherwise(0)) \
    .withColumn('pom',reduce(add, [F.col(x) for x in weather_main_values])) 
df_onehot = df_onehot.withColumn('wm_other', when(df_onehot.pom == 0, 1).otherwise(0)) 
df_onehot = df_onehot.drop(df_onehot.pom) \
    .drop(df_onehot.weather_main)


df = df_agg.join(df_onehot, ['date', 'hour']) 

df = df.drop(df.weather_main) \
    .sort('date', 'hour', ascending=[True, True])

#df.show()
### TARGET VARIABLE ###
tomtom_schema = StructType([
    StructField("iconCategory", StringType(), True),  # Categorical column
    StructField("id", StringType(), True),  # Categorical column
    StructField("probabilityOfOccurrence", StringType(), True),  # Categorical column
    StructField("delay", DoubleType(), True),  # Numerical column
    StructField("magnitudeOfDelay", StringType(), True),  # Categorical column
    StructField("length", DoubleType(), True),  # Numerical column
    StructField("observationTime", TimestampType(), True)  # Categorical column (assuming it's a string)
])

#df_live_tomtom = read_parquets_to_df(folder="tomtom/live", schema=tomtom_schema)
#df_live_tomtom = df_live_tomtom.drop('timeValidity')

df_hist_tomtom = read_parquets_to_df(folder="tomtom/historical", schema=tomtom_schema)
df_hist_tomtom = df_hist_tomtom.drop('UNNAMED_FIELD')
df_hist_tomtom = df_hist_tomtom.drop('timeValidity')

#df = df_hist.unionByName(df_live)
df_tomtom = df_hist_tomtom
df_tomtom = df_tomtom.withColumn('ny_timestamp', from_utc_timestamp(col('observationTime'), 'America/New_York')) \
    .filter(df_tomtom.iconCategory == 6)

df_tomtom_agg = df_tomtom.filter(col("length").isNotNull()) \
    .groupBy('ny_timestamp') \
    .agg(sum("length").alias("length_of_traffic_jams")) \
    .withColumn('date', to_date(df_tomtom.ny_timestamp)) \
    .withColumn('hour', hour(df_tomtom.ny_timestamp))

df_tomtom_agg = df_tomtom_agg.groupBy('date', 'hour') \
    .agg(round(avg("length_of_traffic_jams"),2).alias("avg_length_of_traffic_jams")) \

# Add lagged target variable as it is a time series!!!
windowSpec = Window().orderBy("date", "hour")

df_tomtom_agg = (df_tomtom_agg
                .withColumn("avg_length_of_traffic_jams_1_hour_ago", lag("avg_length_of_traffic_jams",offset=2,default=0).over(windowSpec))
)
# df_tomtom_agg.show()

df = df.join(df_tomtom_agg, ['date', 'hour']) 
result=df.agg(min("date").alias("min_date"),
               max("date").alias("max_date"))
print(f"number of rows:{df.count()}")
result.show(truncate=False)
df = df.drop(df.date)
#df.show()

df_null = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
#df_null.show()

df = df.na.drop("any")
# df_null = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
# df_null.show()


### TRAIN/TEST SPLIT ###

train_df, test_df = df.randomSplit([0.7, 0.3], seed=222)

label_name = "avg_length_of_traffic_jams"

# get a column with feature colums combined
assembler = VectorAssembler(
    inputCols=[x.name for x in train_df.schema if x.name != label_name],
    outputCol="features")
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)


print("train")
train_df.agg(avg(label_name)).show()
print("test")
test_df.agg(avg(label_name)).show()


params={
    "objective":"reg:absoluteerror",
    "max_depth":5,
    "n_estimators":15,
    "min_child_weight":5,
}
            ## MODEL ###
spark_reg_estimator = SparkXGBRegressor(
    features_col='features',
    label_col=label_name,
    tree_method='hist',
    **params
)

#train the model
model = spark_reg_estimator.fit(train_df)


# predict on test data
train_predict_df = model.transform(train_df)
predict_df = model.transform(test_df)
train_predict_df.show()
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=label_name)

print("###")
print(params)
print(train_df.agg(avg(label_name)).show())
print(test_df.agg(avg(label_name)).show())

print(f"The MAE for train {evaluator.evaluate(train_predict_df, {evaluator.metricName: 'mae'})}")
print(f"The MAE for test {evaluator.evaluate(predict_df, {evaluator.metricName: 'mae'})}")
print("###")


# save the model
# model.save("/models/tomtom_xgboost_model")
