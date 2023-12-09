from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.types import *
import os

def mode_result(df, colname):
    count = df.groupBy(['date', 'hour', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour')
            .agg(F.max(F.struct(F.col('count'),
                                F.col(colname))).alias('max'))
            .select(F.col('date'), F.col('hour'), F.col(f'max.{colname}').alias(f'mode_{colname}'))
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

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

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
    

#df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather/live")

df_live = read_parquets_to_df(folder="openweather/live", schema=weather_schema)
df_live = df_live.drop('wind_deg')
#df_live.show()


df_hist = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
df_hist = df_hist.drop('UNNAMED_FIELD')
#df_hist.show()

df = df_hist.unionByName(df_live)
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp))
#df.show()

df_agg = df.groupBy("date", "hour") \
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
weather_description_result = mode_result(df, 'weather_description')

df_agg = df_agg.join(weather_main_result, ['date', 'hour']).join(weather_description_result, ['date', 'hour']).sort("date", "hour", ascending=[True, True])
df_agg.show()
print(df_agg.select('date').distinct().collect())