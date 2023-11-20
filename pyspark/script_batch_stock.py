from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()


df = spark.read.option("recursiveFileLookup", "true").parquet("/stock")
#df = df.withColumn('timestamp', df.timestamp.cast(dataType=TimestampType()))
df = df.withColumn('date', to_date(df.datetime))
df = df.withColumn('hour', hour(df.datetime))
df_agg = df.groupBy("date", "hour") \
    .agg(avg("transactions").alias("avg_transactions"),
        avg("volume").alias("avg_volume"), 
         avg("open").alias("avg_open"),
         avg("high").alias("avg_high"),
         avg("low").alias("avg_low"),
         avg("open").alias("avg_open"),
         avg("vwap").alias("avg_vwap"))


df_agg.show()