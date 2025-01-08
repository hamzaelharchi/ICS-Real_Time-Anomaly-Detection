spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  --jars kafka-clients-3.5.0.jar \
  --driver-class-path kafka-clients-3.5.0.jar \
  --conf "spark.executor.extraClassPath=kafka-clients-3.5.0.jar" \
  work/pyspark-read.py