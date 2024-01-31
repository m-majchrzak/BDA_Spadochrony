from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os

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

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

df_live = read_parquets_to_df(folder="stock/live", schema=stock_schema)
#df_live.show()

df_hist = read_parquets_to_df(folder="stock/historical", schema=stock_schema)
df_hist = df_hist.drop('UNNAMED_FIELD')
#df_hist.show()

df = df_hist.unionByName(df_live)
#df = df_live
df.show()
df = df.withColumn('timestamp_from_datetime', df.datetime.cast(dataType=TimestampType()))
df = df.withColumn('ny_timestamp', from_utc_timestamp(col('timestamp_from_datetime'), 'America/New_York'))
df = df.withColumn("corrected_timestamp",col('ny_timestamp') - expr("INTERVAL 20 minutes"))
df = df.withColumn('date', to_date(df.corrected_timestamp))
df = df.withColumn('hour', hour(df.corrected_timestamp))

df_agg = df.groupBy("date", "hour") \
    .agg(count("transactions").alias("count"),
        round(avg("transactions"),2).alias("avg_transactions"),
        round(avg("volume"),2).alias("avg_volume"), 
        round(avg("open"),2).alias("avg_open"),
        round(avg("high"),2).alias("avg_high"),
        round(avg("low"),2).alias("avg_low"),
        round(avg("open"),2).alias("avg_open"),
        round(avg("vwap"),2).alias("avg_vwap")) \
    .sort("date", "hour", ascending=[True, True])


df_agg.show()
#df_agg.show(n=df_agg.count(), truncate = False)
print(f"The dataframe has {df_agg.count()} rows.")