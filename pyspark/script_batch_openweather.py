from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.types import *
import os
from google.cloud import bigtable
import datetime
import math

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
#df_live = df_live.drop('wind_deg')
# df_live.show()


df_hist = read_parquets_to_df(folder="openweather/historical", schema=weather_schema)
#df_hist = df_hist.drop('UNNAMED_FIELD')
#df_hist.show()

if df_live == None:
    df = df_hist
else:
    df = df_hist.unionByName(df_live)

#df.show()
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp))
#df.show()

#df_agg = df.groupBy("date") \
df_agg = df.groupBy("date", "hour") \
    .agg(round(avg("temp"),2).alias("avg_temp"), \
         round(avg("visibility"),2).alias("avg_visibility"), \
         round(avg("pressure"),2).alias("avg_pressure"), \
         round(avg("clouds"),2).alias("avg_clouds"), \
         round(avg("feels_like"),2).alias("avg_feels_like"), \
         round(avg("temp_max"),2).alias("avg_temp_max"), \
         round(avg("temp_min"),2).alias("avg_temp_min"), \
         round(avg("humidity"),2).alias("avg_humidity"), \
         round(avg("wind_speed"),2).alias("avg_wind_speed")) 

weather_main_result = mode_result(df, 'weather_main')
weather_description_result = mode_result(df, 'weather_description')

df_agg = df_agg.join(weather_main_result, ['date', 'hour']).join(weather_description_result, ['date', 'hour']).sort("date", "hour", ascending=[True, True])

df_agg.show()
#df_agg.show(n=df_agg.count(), truncate = False)
print(f"The dataframe has {df_agg.count()} rows.")


### WRITING TO BIGTABLE

client = bigtable.Client(project="bda-project-412623", admin=True)
instance = client.instance("bda-bigtable")
table = instance.table("batch_openweather")
timestamp = datetime.datetime.utcnow()

df_len = df_agg.count()
row_list = df_agg.collect()

time_columns = ["date", "hour"]
weather_avg_columns = ["avg_temp", "avg_pressure", "avg_clouds", "avg_clouds", "avg_feels_like", "avg_temp_max", "avg_temp_min", "avg_humidity", "avg_wind_speed"]
weather_mode_colums = ["mode_weather_main", "mode_weather_description"]


batch_size = 5000
no_batches = math.ceil(df_len / batch_size)
for batch in range(no_batches):
    if batch == 0:
        start = 0
        end = batch_size
    elif batch == no_batches - 1:
        start += batch_size
        end = df_len
    else:
        start+=batch_size
        end+=batch_size
    row_names = [str(row_list[i].__getitem__('date')) + '_' + str(row_list[i].__getitem__('hour')) for i in range(start, end)]
    rows = [table.direct_row(row_name) for row_name in row_names]
    for i in range(end-start):
        for column in time_columns:
            rows[i].set_cell("time", column, str(row_list[start+i].__getitem__(column)), timestamp)
        for column in weather_avg_columns:
            rows[i].set_cell("weather_avg", column, str(row_list[start+i].__getitem__(column)), timestamp)
        for column in weather_mode_colums:
            rows[i].set_cell("weather_mode", column, str(row_list[start+i].__getitem__(column)), timestamp)
    response = table.mutate_rows(rows)
    for i, status in enumerate(response):
        if status.code != 0:
            print("Error writing row: {}".format(status.message))

    print("Successfully wrote {} rows.".format(df_len))




