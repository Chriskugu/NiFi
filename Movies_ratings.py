from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Create Spark Session
spark = SparkSession.builder \
    .appName("Movie Rating Prediction") \
    .getOrCreate()

# Load CSV from HDFS
movies_df = spark.read.csv(
    "hdfs:///Movies_ratings.csv",
    header=True,
    inferSchema=True
)

# Show schema
movies_df.printSchema()

# Remove nulls
movies_df = movies_df.na.drop()

# Features
assembler = VectorAssembler(
    inputCols=["user", "movies"],
    outputCol="features"
)

data = assembler.transform(movies_df)

final_df = data.select("features", "rating")

# Split data
train_data, test_data = final_df.randomSplit([0.7, 0.3], seed=42)

# Train model
lr = LinearRegression(
    featuresCol="features",
    labelCol="rating"
)

model = lr.fit(train_data)

# Predictions
predictions = model.transform(test_data)

predictions.select(
    "rating",
    "prediction"
).show(20, False)

# Evaluate
results = model.evaluate(test_data)

rmse = results.rootMeanSquaredError
r2 = results.r2

print("================================")
print(f"RMSE = {rmse}")
print(f"R2   = {r2}")
print("================================")

# Save metrics to HBase
data = [
    ('movie_metrics', 'cf:rmse', str(rmse)),
    ('movie_metrics', 'cf:r2', str(r2))
]

def write_to_hbase_partition(partition):

    connection = happybase.Connection('master')
    connection.open()

    table = connection.table('my_table')

    for row in partition:
        row_key, column, value = row

        table.put(
            row_key,
            {column: value}
        )

    connection.close()

spark.sparkContext.parallelize(data) \
    .foreachPartition(write_to_hbase_partition)

spark.stop()
