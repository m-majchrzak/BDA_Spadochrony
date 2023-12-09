from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
import os
from operator import add
from functools import reduce
from xgboost.spark import SparkXGBRegressor

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

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

# weatherSchema = StructType([
#     StructField("temp", DoubleType(), True),
#     StructField("weather_description", StringType(), True),
#     StructField("visibility", IntegerType(), True),
#     StructField("pressure", IntegerType(), True),
#     StructField("clouds", IntegerType(), True),
#     StructField("feels_like", DoubleType(), True),
#     StructField("temp_max", DoubleType(), True),
#     StructField("weather_main", StringType(), True),
#     StructField("temp_min", DoubleType(), True),
#     StructField("wind_deg", StringType(), True),
#     StructField("humidity", IntegerType(), True),
#     StructField("wind_speed", DoubleType(), True),
#     StructField("timestamp", IntegerType(), True),
#     ])

## WEATHER
# reading all files from /openweather
# unioned_df = None

# base_path = "/openweather"
# glob_pattern = "*"
# jvm = spark.sparkContext._jvm
# fs_root = jvm.java.net.URI.create(base_path)
# conf = spark.sparkContext._jsc.hadoopConfiguration()
# fs = jvm.org.apache.hadoop.fs.FileSystem.get(fs_root, conf)
# path_glob = jvm.org.apache.hadoop.fs.Path(os.path.join(base_path, glob_pattern))
# status_list = fs.globStatus(path_glob)
# for status in status_list:
# 	raw_path = status.getPath().toUri().getRawPath()
# 	df = spark.read.parquet(raw_path)
# 	df = spark.createDataFrame(df.rdd, schema=weatherSchema)
# 	if not unioned_df:
# 		unioned_df = df
# 	else:
# 		unioned_df = unioned_df.union(df)
	

df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather") 

df = df.withColumn('timestamp', df.timestamp.cast(dataType=TimestampType()))
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp)) 
df = df.withColumn('day_of_week', dayofweek(df.timestamp)) \
    .withColumn('month', month(df.timestamp)) \
    #.withColumn('min_since_midnight', hour(df.timestamp)*24+minute(df.timestamp)) \
df = df.drop(df.weather_description) \
    .drop(df.UNNAMED_FIELD)

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
weather_main_values = ['thunderstorm',
                       'drizzle',
                       'rain',
                       'snow',
                       'clear', 
                       'clouds']
df_onehot = df_onehot.withColumn('thunderstorm', when(df_onehot.weather_main == 'Thunderstorm', 1).otherwise(0)) \
    .withColumn('drizzle', when(df_onehot.weather_main == 'Drizzle', 1).otherwise(0)) \
    .withColumn('rain', when(df_onehot.weather_main == 'Rain', 1).otherwise(0)) \
    .withColumn('snow', when(df_onehot.weather_main == 'Snow', 1).otherwise(0)) \
    .withColumn('clear', when(df_onehot.weather_main == 'Clear', 1).otherwise(0)) \
    .withColumn('clouds', when(df_onehot.weather_main == 'Clouds', 1).otherwise(0)) \
    .withColumn('pom',reduce(add, [F.col(x) for x in weather_main_values])) 
df_onehot = df_onehot.withColumn('other', when(df_onehot.pom == 0, 1).otherwise(0)) \
    .drop(df_onehot.pom) \
    .drop(df_onehot.weather_main)

df = df_agg.join(df_onehot, ['date', 'hour']) \
    .drop(df.weather_main) \
    .sort('date', 'hour', ascending=[True, True])

### TARGET VARIABLE ###
df_tomtom = spark.read.option("recursiveFileLookup", "true").parquet("/tomtom")
df_tomtom = df_tomtom.withColumn('timestamp', df_tomtom.observationTime.cast(dataType=TimestampType())) \
    .filter(df_tomtom.iconCategory == 6)

df_tomtom_agg = df_tomtom.filter(col("length").isNotNull()) \
    .groupBy('timestamp') \
    .agg(sum("length").alias("length_of_traffic_jams")) \
    .withColumn('date', to_date(df_tomtom.timestamp)) \
    .withColumn('hour', hour(df_tomtom.timestamp))

df_tomtom_agg = df_tomtom_agg.groupBy('date', 'hour') \
    .agg(avg("length_of_traffic_jams").alias("avg_length_of_traffic_jams")) \

df = df.join(df_tomtom_agg, ['date', 'hour']) \
    .drop(df.date) \
    .drop(df.weather_main)

### TRAIN/TEST SPLIT ###
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)


# assume the label column is named "class"
label_name = "avg_length_of_traffic_jams"

# get a list with feature column names
feature_names = [x.name for x in train_df.schema if x.name != label_name]

### MODEL ###
spark_reg_estimator = SparkXGBRegressor(
  features_col=feature_names,
  label_col=label_name,
  num_workers=2,
)

model = spark_reg_estimator.fit(df)

# predict on test data
predict_df = model.transform(test_df)
predict_df.show()


# save and use later
# regressor = SparkXGBRegressor()
# model = regressor.fit(train_df)
# # save the model
# model.save("/tmp/xgboost-pyspark-model")
# # load the model
# model2 = SparkXGBRankerModel.load("/tmp/xgboost-pyspark-model")


