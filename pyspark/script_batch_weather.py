from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()


df = spark.read.option("recursiveFileLookup", "true").parquet("/openweather")
df = df.withColumn('timestamp', df.timestamp.cast(dataType=TimestampType()))
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp))
df_agg = df.groupBy("date", "hour") \
    .agg(avg("temp").alias("avg_temp"), \
         #mode("weather_description").alias("mode_weather_description"), \
         avg("visibility").alias("avg_visibility"), \
         avg("pressure").alias("avg_pressure"), \
         avg("clouds").alias("avg_clouds"), \
         avg("feels_like").alias("avg_feels_like"), \
         avg("temp_max").alias("avg_temp_max"), \
         #mode("weather_main").alias("mode_weather_main"), \
         avg("temp_min").alias("avg_temp_min"), \
         avg("humidity").alias("avg_humidity"), \
         avg("wind_speed").alias("avg_wind_speed")) \

#mode = df.groupby("weather_id").count().orderBy("count", ascending=False).first()[0]

df_agg.show()
