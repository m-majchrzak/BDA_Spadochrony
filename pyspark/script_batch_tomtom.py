from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *


spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()


df = spark.read.option("recursiveFileLookup", "true").parquet("/tomtom")
df = df.withColumn('timestamp', df.observationTime.cast(dataType=TimestampType()))
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp))
df_agg = df.groupBy("date", "hour", "category") \
    .agg(#mode("probabilityOfOccurrence").alias("probabilityOfOccurrence"), \
        avg("delay").alias("avg_delay"), \
        avg("length").alias("avg_length"), \
        #mode("probabilityOfOccurrence").alias("probabilityOfOccurrence"), \
         ) 

df.show()
df_agg.show()