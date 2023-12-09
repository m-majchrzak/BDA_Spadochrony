from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import os
from operator import add
from functools import reduce

def mode_result(df, colname):
    count = df.groupBy(['date', 'hour', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour')
            .agg(F.max(F.struct(F.col('count'),
                                F.col(colname))).alias('max'))
            .select(F.col('date'), F.col('hour'), F.col(f'max.{colname}').alias(f'mode_{colname}'))
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
	

# alteratywna metoda wczytywania - działa jeśli wszystkie parquety mają taki sam scheme
df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather") 
df = df.withColumn('timestamp', df.timestamp.cast(dataType=TimestampType()))
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp)) 
df = df.withColumn('day_of_week', dayofweek(df.timestamp)) \
    .withColumn('month', month(df.timestamp)) \
    .withColumn('min_since_midnight', hour(df.timestamp)*24+minute(df.timestamp)) \
    .drop(df.weather_description) \
    .drop(df.UNNAMED_FIELD)
#df.show()

df_onehot = df.select(col('timestamp'), col('weather_main')) 
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

df = df.join(df_onehot, ['timestamp']) \
    .drop(df.weather_main) \
    .drop(df.timestamp) \
    .sort("date", "hour", ascending=[True, True])
df.show()

### TARGET VARIABLE
df_stock = spark.read.option("recursiveFileLookup", "true").parquet("/stock")
df_stock = df_stock.withColumn('date', to_date(df_stock.datetime))
df_stock = df_stock.withColumn('hour', hour(df_stock.datetime))
df_stock_agg = df_stock.groupBy("date", "hour") \
    .agg(count("transactions").alias("count")) \
    .sort("date", "hour", ascending=[True, True])

df = df.join(df_stock_agg, ['date', 'hour']) \
    .drop(df.date) \
    .drop(df.hour)