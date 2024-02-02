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

def hdfs_delete(path):
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    fs.delete(spark._jvm.org.apache.hadoop.fs.Path(path), True)
    return True

def hdfs_delete_files_from_dir(folder):
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
        hdfs_delete(raw_path)

tomtom_schema = StructType([
    StructField("iconCategory", StringType(), True),  # Categorical column
    StructField("id", StringType(), True),  # Categorical column
    StructField("probabilityOfOccurrence", StringType(), True),  # Categorical column
    StructField("delay", DoubleType(), True),  # Numerical column
    StructField("magnitudeOfDelay", StringType(), True),  # Categorical column
    StructField("length", DoubleType(), True),  # Numerical column
    StructField("observationTime", TimestampType(), True)  # Categorical column (assuming it's a string)
])

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

# CHECK NUMBER OF ROWS IN HISTORICAL

# df_historical = read_parquets_to_df(folder="tomtom/historical", schema=tomtom_schema)
# print(df_historical.count())

### READ FROM LIVE ###
df_live = read_parquets_to_df(folder="tomtom/live", schema=tomtom_schema)
#df_live.show()

### WRITE TO HITORICAL ###

data=[["1"]]
data_df=spark.createDataFrame(data,["id"])
data_df = data_df.withColumn("date", date_format(current_date(), format="MM-dd-yyyy"))
curr_date = data_df.first()['date']
df_live.write.parquet(f"/tomtom/historical/tomtom-{curr_date}.parquet", mode="overwrite")

# CHECK NUMBER OF ROWS IN HISTORICAL

# df_historical = read_parquets_to_df(folder="tomtom/historical", schema=tomtom_schema)
# print(df_historical.count())

### DELETE FROM LIVE ###
hdfs_delete_files_from_dir(folder="tomtom/live")