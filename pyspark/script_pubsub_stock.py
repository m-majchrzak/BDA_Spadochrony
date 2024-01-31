from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

project_number = 684093064430
location = "europe-central2" 
subscription_id = "stock-spark"

spark = SparkSession.builder.appName("read-app").master("yarn").getOrCreate()

sdf = (
    spark.readStream.format("pubsublite")
    .option(
        "pubsublite.subscription",
        f"projects/{project_number}/locations/{location}/subscriptions/{subscription_id}",
    )
    .load()
)

sdf = sdf.withColumn("data", sdf.data.cast(StringType()))

query = (
    sdf.writeStream.format("console")
    .outputMode("append")
    .trigger(processingTime="1 second")
    .start()
)

# Wait 120 seconds (must be >= 60 seconds) to start receiving messages.
query.awaitTermination(120)
query.stop()