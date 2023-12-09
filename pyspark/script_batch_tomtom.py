from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.types import *

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


spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()


df = spark.read.option("recursiveFileLookup", "true").parquet("/tomtom")

# Columns:
# - iconCategory - cat
# - id - cat
# - probabilityOfOccurence - cat
# - delay - numerical
# - magnitudeOfDelay - categorical
# - length - numerical
# - observationTime

df = df.withColumn('timestamp', df.observationTime.cast(dataType=TimestampType()))
df = df.withColumn('date', to_date(df.timestamp))
df = df.withColumn('hour', hour(df.timestamp))

df.show()

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