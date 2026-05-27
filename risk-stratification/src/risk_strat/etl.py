from __future__ import annotations
def get_pyspark_etl_example():
    return """# PySpark ETL Pipeline Example
# Run with: spark-submit etl_pipeline.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, trim
spark = SparkSession.builder.appName("readmission_etl").getOrCreate()
# Load raw EHR data
raw_df = spark.read.option("inferSchema", "true").csv("s3://health-data/records/*.csv")
# Clean column names
cleaned_df = raw_df.select([trim(col(c)).alias(c) for c in raw_df.columns])
# Handle missing values
from pyspark.ml.feature import Imputer
numeric_cols = [field.name for field in raw_df.schema if field.dataType.typeName() in ["integer", "double"]]
imputer = Imputer(strategy="median", inputCols=numeric_cols, outputCols=numeric_cols)
imputed_df = imputer.fit(cleaned_df).transform(cleaned_df)
# Target mapping
readmission_df = imputed_df.withColumn(
    "readmitted_binary",
    when(col("readmitted") == "<30", 1).otherwise(0)
)
# Write to data warehouse
readmission_df.write.mode("overwrite").parquet("s3://health-data/ready/readmission_features")
print("ETL complete!")
"""
def get_etl_config_example():
    return {
        "sources": {
            "ehr": {
                "type": "sql",
                "connection": "postgresql://ehr-prod:5432/hospital",
                "query": "SELECT * FROM encounters WHERE admission_date >= CURRENT_DATE - INTERVAL 90 DAY",
            },
            "labs": {"type": "parquet", "path": "s3://health-data/labs/"},
        },
        "transformations": [
            {"name": "drop_identifiers", "columns": ["ssn", "mrn"]},
            {"name": "handle_missing", "strategy": "median"},
            {"name": "normalize_dates", "format": "YYYY-MM-DD"},
        ],
        "validation": {
            "row_count_min": 50000,
            "null_percentage_max": 0.3,
            "schema_check": True,
        },
        "output": {
            "type": "parquet",
            "path": "s3://health-data/ready/train_data/",
            "partitions": ["year", "month"],
        },
    }
