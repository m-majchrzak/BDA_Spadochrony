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
            .agg(F.max(F.struct(F.col('count'),
                                F.col(colname))).alias('max'))
            .select(F.col('date'), F.col('hour'), F.col(f'max.{colname}').alias(f'{colname}'))
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

#df_live = read_parquets_to_df(folder="openweather/live", schema=weather_schema)
#df_live = df_live.drop('wind_deg')
#df_live.show()


df_hist = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
df_hist = df_hist.drop('UNNAMED_FIELD')
#df_hist.show()

#df = df_hist.unionByName(df_live)
df = df_hist


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
stock_schema = StructType([
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
    StructField("datetime", TimestampType(), True)
])

df_hist_stock = read_parquets_to_df(folder="stock/historical", schema=stock_schema)
df_hist_stock = df_hist_stock.drop('UNNAMED_FIELD')

#df_live_stock = read_parquets_to_df(folder="stock/live", schema=stock_schema)
#df_stock = df_hist_stock.unionByName(df_live_stock)
df_stock = df_hist_stock

df_stock = df_stock.withColumn('timestamp_from_datetime', df_stock.datetime.cast(dataType=TimestampType()))
df_stock = df_stock.withColumn("corrected_timestamp",col("timestamp_from_datetime") - expr("INTERVAL 20 minutes"))
df_stock = df_stock.withColumn('date', to_date(df_stock.corrected_timestamp))
df_stock = df_stock.withColumn('hour', hour(df_stock.corrected_timestamp))
df_stock_agg = df_stock.groupBy("date", "hour") \
    .agg(count("transactions").alias("number_of_transactions")) \
    .sort("date", "hour", ascending=[True, True])

df = df.join(df_stock_agg, ['date', 'hour']) \
    .drop(df.date) \
    .drop(df.hour)

#df.show()
### TRAIN/TEST SPLIT ###

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)


# assume the label column is named "class"
label_name = "number_of_transactions"

# get a list with feature column names
feature_names = [x.name for x in train_df.schema if x.name != label_name]
features = train_df.select(*feature_names)
features.show()

### MODEL ###
spark_reg_estimator = SparkXGBRegressor(
  features_col=features,
  label_col=label_name,
  num_workers=2
)

model = spark_reg_estimator.fit(df)

# predict on test data
predict_df = model.transform(test_df)
predict_df.show()