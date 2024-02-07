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

def mode_result_by_category_and_id(df, colname):
    count = df.groupBy(['date', 'hour', 'iconCategory', 'id', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour', 'iconCategory', 'id')
            .agg(max(struct(col('count'),
                                col(colname))).alias('max'))
            .select(col('date'), col('hour'), col('iconCategory'), col('id'), col(f'max.{colname}').alias(f'mode_{colname}'))
            )
    return result

def mode_result_by_category(df, colname):
    count = df.groupBy(['date', 'hour', 'iconCategory', colname]).count().alias('counts')
    result = (count
            .groupBy('date', 'hour', 'iconCategory')
            .agg(max(struct(col('count'),
                                col(colname))).alias('max'))
            .select(col('date'), col('hour'), col('iconCategory'), col(f'max.{colname}').alias(f'{colname}'))
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

tomtom_schema = StructType([
    StructField("iconCategory", StringType(), True),  # Categorical column
    StructField("id", StringType(), True),  # Categorical column
    StructField("probabilityOfOccurrence", StringType(), True),  # Categorical column
    StructField("delay", DoubleType(), True),  # Numerical column
    StructField("magnitudeOfDelay", StringType(), True),  # Categorical column
    StructField("length", DoubleType(), True),  # Numerical column
    StructField("observationTime", TimestampType(), True)  # Categorical column (assuming it's a string)
])

df_live = read_parquets_to_df(folder="tomtom/live", schema=tomtom_schema)
# df_live = df_live.drop('timeValidity')

df_hist = read_parquets_to_df(folder="tomtom/historical", schema=tomtom_schema)
# df_hist = df_hist.drop('UNNAMED_FIELD')
# df_hist = df_hist.drop('timeValidity')

# if df_live == None:
#     df = df_hist
# else:
#     df = df_hist.unionByName(df_live)
    
df = df_live
    
df = df.withColumn('ny_timestamp', from_utc_timestamp(col('observationTime'), 'America/New_York'))
df = df.withColumn('date', to_date(df.ny_timestamp))
df = df.withColumn('hour', hour(df.ny_timestamp))

#df.show()

# remove nulls before calculating the mean

df_agg_by_id_delay = df.filter(col("delay").isNotNull()) \
    .groupBy('date', 'hour', 'iconCategory', 'id') \
    .agg(avg("delay").alias("avg_delay"))

df_agg_by_id_length = df.filter(col("length").isNotNull()) \
    .groupBy('date', 'hour', 'iconCategory', 'id') \
    .agg(avg("length").alias("avg_length"))

df_agg_by_id = df.groupBy('date', 'hour', 'iconCategory', 'id') \
    .agg(count("probabilityOfOccurrence").alias('count')) \
    .join(df_agg_by_id_delay, ['date', 'hour', 'iconCategory', 'id']) \
    .join(df_agg_by_id_length, ['date', 'hour', 'iconCategory', 'id']) \
    .sort("date", "hour", ascending=[True, True])


probabilityOfOccurence_result_by_category_and_id = mode_result_by_category_and_id(df, 'probabilityOfOccurrence')
magnitudeOfDelay_result_by_category_and_id = mode_result_by_category_and_id(df, 'magnitudeOfDelay')
df_agg_by_id = df_agg_by_id.join(probabilityOfOccurence_result_by_category_and_id, ['date', 'hour', 'iconCategory', 'id']).join(magnitudeOfDelay_result_by_category_and_id, ['date', 'hour', 'iconCategory', 'id']).sort("date", "hour", ascending=[True, True])


df_agg = df_agg_by_id.groupBy('date', 'hour', 'iconCategory') \
    .agg(count("id").alias('count_id'),
         round(avg("avg_delay"),2).alias("avg_delay"), \
         round(avg("avg_length"),2).alias("avg_length"))

probabilityOfOccurence_result_by_category = mode_result_by_category(df_agg_by_id, 'mode_probabilityOfOccurrence')
magnitudeOfDelay_result_by_category= mode_result_by_category(df_agg_by_id, 'mode_magnitudeOfDelay')
df_agg = df_agg.join(probabilityOfOccurence_result_by_category, ['date', 'hour', 'iconCategory']).join(magnitudeOfDelay_result_by_category, ['date', 'hour', 'iconCategory']).sort("date", "hour", ascending=[True, True])

# ew ToDo - uzupełnianie 0 tam, gdzie nie było żadnego wystąpienia zdarzenia danego typu w danej godzinie

df_agg.show()
#df_agg.show(n=df_agg.count(), truncate = False) 
print(f"The dataframe has {df_agg.count()} rows.")

### WRITING TO BIGTABLE

client = bigtable.Client(project="grand-harbor-413313", admin=True)
instance = client.instance("bda-bigtable")
table = instance.table("batch_tomtom")
timestamp = datetime.datetime.utcnow()

df_len = df_agg.count()
row_list = df_agg.collect()

time_columns = ["date", "hour"]
tomtom_columns = ["iconCategory", "count_id", "avg_delay", "avg_length", "mode_probabilityOfOccurrence", "mode_magnitudeOfDelay"]

batch_size = 5000
no_batches = math.ceil(df_len / batch_size)
for batch in range(no_batches):
    if batch == 0:
        start = 0
        if no_batches == 1:
            end = df_len
        else:
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
        for column in tomtom_columns:
            rows[i].set_cell("tomtom", column, str(row_list[start+i].__getitem__(column)), timestamp)
    response = table.mutate_rows(rows)
    for i, status in enumerate(response):
        if status.code != 0:
            print("Error writing row: {}".format(status.message))

    print("Successfully wrote {} rows.".format(end-start))