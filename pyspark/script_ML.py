from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import os


spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

weatherSchema = StructType([
    StructField("temp", DoubleType(), True),
    StructField("weather_description", StringType(), True),
    StructField("visibility", IntegerType(), True),
    StructField("pressure", IntegerType(), True),
    StructField("clouds", IntegerType(), True),
    StructField("feels_like", DoubleType(), True),
    StructField("temp_max", DoubleType(), True),
    StructField("weather_main", StringType(), True),
    StructField("temp_min", DoubleType(), True),
    StructField("wind_deg", StringType(), True),
    StructField("humidity", IntegerType(), True),
    StructField("wind_speed", DoubleType(), True),
    StructField("timestamp", IntegerType(), True),
    ])

## WEATHER
# reading all files from /openweather
unioned_df = None

base_path = "/openweather"
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
	df = spark.createDataFrame(df.rdd, schema=weatherSchema)
	if not unioned_df:
		unioned_df = df
	else:
		unioned_df = unioned_df.union(df)
	

# alteratywna metoda wczytywania - działa jeśli wszystkie parquety mają taki sam scheme
#df = spark.read.schema(weatherSchema).option("recursiveFileLookup", "true").parquet("/openweather/live")

df = unioned_df
df = df.withColumn('timestamp', df.timestamp.cast(dataType=TimestampType()))
df = df.withColumn('dayofweek', dayofweek(df.timestamp))
df = df.withColumn('month', month(df.timestamp))
df = df.withColumn('min_since_midnight', hour(df.timestamp)*24+minute(df.timestamp))
df = df.drop(df.weather_description)
df = df.drop(df.timestamp)
df.show()

# one-hot encoding - ToDo
#indexer = StringIndexer(inputCol='weather_main', outputCol='weather_numeric')
#df_indexed = indexer.fit(df).transform(df)
#ohe = OneHotEncoder(inputCol="weather_numeric", outputCol="weather_onehot")
#df_onehot = ohe.fit(df_indexed).transform(df_indexed)
#df_col_onehot = df_onehot.select('*', vector_to_array('weather_onehot').alias('col_onehot'))
#df_col_onehot = df_onehot.select('*', vector_to_array('class_onehot').alias('col_onehot'))
#cols_expanded = [(F.col('col_onehot')[i]) for i in range(num_categories)]
#df_cols_onehot = df_col_onehot.select('name', 'class', *cols_expanded)
#df_onehot.show()

