from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

### LOOK AT HISTORICAL ###
historical_df = spark.read.option("recursiveFileLookup", "true").parquet("/tomtom/historical")
#historical_df.show()

### READ FROM LIVE ##
df = spark.read.option("recursiveFileLookup", "true").parquet("/tomtom/live")
#df.show()

### WRITE TO HITORICAL ###
#curr_date = date_format(current_timestamp(),).alias("yyyy-MM-dd")
data=[["1"]]
data_df=spark.createDataFrame(data,["id"])
data_df = data_df.withColumn("date", date_format(current_date(),"MM-dd-yyyy"))
curr_date = data_df.first()['date']
df.write.mode('overwrite').parquet(f"/tomtom/historical/tomtom-{curr_date}.parquet")

df_read = spark.read.parquet(f"/tomtom/historical/tomtom-{curr_date}.parquet")
df_read.show()