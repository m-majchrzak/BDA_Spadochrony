from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

### LOOK AT HISTORICAL ###
historical_df = spark.read.option("recursiveFileLookup", "true").parquet("/stock/historical")
historical_df.show()

### READ FROM LIVE ###
df = spark.read.option("recursiveFileLookup", "true").parquet("/stock/live")
df.show()

### WRITE TO HITORICAL ###
data=[["1"]]
data_df=spark.createDataFrame(data,["id"])
data_df = data_df.withColumn("date", date_format(current_date(),"MM-dd-yyyy"))
curr_date = data_df.first()['date']

df.write.parquet(f"/stock/historical/stock-{curr_date}.parquet")

df_read = spark.read.parquet(f"/stock/historical/stock-{curr_date}.parquet")
df_read.show()