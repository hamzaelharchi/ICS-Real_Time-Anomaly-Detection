from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from collections import deque
from kafka import KafkaProducer
import traceback

# Initialize Spark
sc = SparkContext('local')
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")

# Kafka configuration
INPUT_TOPIC = 'swat'
OUTPUT_TOPIC = 'swat-anomaly'
BOOTSTRAP_SERVERS = "kafka:29092"

# Model configuration
WINDOW_SIZE = 12
DATA_BUFFER_SIZE = WINDOW_SIZE

# Global buffer for accumulating data points
data_buffer = {
    'full_data': deque(maxlen=DATA_BUFFER_SIZE),
    'numeric_data': deque(maxlen=DATA_BUFFER_SIZE)
}

def load_models_and_config():
    """Load ONNX models, scaler, and thresholds."""
    try:
        encoder_session = ort.InferenceSession("work/encoder.onnx")
        decoder1_session = ort.InferenceSession("work/decoder1.onnx")
        decoder2_session = ort.InferenceSession("work/decoder2.onnx")
        
        with open("work/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        with open("work/thresholds.json", "r") as f:
            thresholds = eval(json.load(f))
            
        return encoder_session, decoder1_session, decoder2_session, scaler, thresholds
    except Exception as e:
        print(f"Error loading models and configurations: {e}")
        raise

def process_data(df, scaler):
    """Preprocess the dataframe by scaling its numeric values."""
    df = df.applymap(lambda x: str(x).replace(",", ".")).astype(float)
    scaled_values = scaler.transform(df)
    return pd.DataFrame(scaled_values, columns=df.columns)

def test_model(encoder_session, decoder1_session, decoder2_session, test_data, alpha=0.5, beta=0.5):
    """Test the model on the test data and return the anomaly scores."""
    results = []
    for batch in test_data:
        batch = batch.reshape(1, -1) if batch.ndim == 1 else batch

        # Run encoder
        z = encoder_session.run(None, {encoder_session.get_inputs()[0].name: batch.astype(np.float32)})[0]

        # Run decoder1 and re-encode
        w1 = decoder1_session.run(None, {decoder1_session.get_inputs()[0].name: z.astype(np.float32)})[0]
        z_w1 = encoder_session.run(None, {encoder_session.get_inputs()[0].name: w1.astype(np.float32)})[0]

        # Run decoder2
        w2 = decoder2_session.run(None, {decoder2_session.get_inputs()[0].name: z_w1.astype(np.float32)})[0]

        # Compute anomaly score
        score = (alpha * np.mean((batch - w1) ** 2, axis=1) + 
                 beta * np.mean((batch - w2) ** 2, axis=1))
        results.append(score)

    return np.array(results)

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    key_serializer=lambda k: str(k).encode('utf-8'),
    value_serializer=lambda v: v.encode('utf-8')
)

# Load models and configurations
encoder_session, decoder1_session, decoder2_session, scaler, thresholds = load_models_and_config()


def process_batch(batch_df, batch_id):
    """Process each micro-batch of data from Kafka."""
    try:
        # Convert Spark DataFrame to Pandas
        rows = batch_df.toPandas()
        
        # Parse JSON for each row
        parsed_rows = rows['value'].apply(json.loads)
        
        for row_data in parsed_rows:
            # Store the full data point
            data_buffer['full_data'].append(row_data)
            
            # Prepare numeric data
            numeric_data = pd.DataFrame([row_data])
            if "Timestamp" in numeric_data:
                numeric_data = numeric_data.drop("Timestamp", axis=1)
            if "Normal/Attack" in numeric_data:
                numeric_data = numeric_data.drop("Normal/Attack", axis=1)
            
            # Scale the data
            scaled_data = process_data(numeric_data, scaler)
            data_buffer['numeric_data'].append(scaled_data.values[0])
            
            # Check if buffer is full
            if len(data_buffer['numeric_data']) == WINDOW_SIZE:
                # Create window for testing
                test_window = np.array(list(data_buffer['numeric_data'])).reshape(1, -1)
                
                # Get anomaly score
                anomaly_score = test_model(encoder_session, decoder1_session, decoder2_session, test_window)[0]
                
                # Determine if anomaly based on threshold
                is_anomaly = int(anomaly_score > thresholds["threshold_youden"])
                
                # Prepare output data
                output_data = data_buffer['full_data'][-1].copy()
                output_data['anomaly_score'] = float(anomaly_score)  # Add the anomaly score
                output_data['is_anomaly'] = is_anomaly
                
                # Send to Kafka
                key = f"anomaly-{batch_id}"
                value = json.dumps(output_data)

                print(value)
                
                print(f"Sending prediction - Anomaly Score: {anomaly_score}, Is Anomaly: {is_anomaly}")
                producer.send(OUTPUT_TOPIC, key=key, value=value)
                
                # Remove oldest entry to slide the window
                data_buffer['numeric_data'].popleft()
                data_buffer['full_data'].popleft()
                
    except Exception as e:
        print(f"Error processing batch: {e}")
        traceback.print_exc()

# Read from Kafka with Structured Streaming
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS) \
    .option("subscribe", INPUT_TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# Parse the JSON data from Kafka
df_parsed = df.selectExpr("CAST(value AS STRING) as value")

# Write stream using foreachBatch
query = df_parsed \
    .writeStream \
    .outputMode("update") \
    .foreachBatch(process_batch) \
    .start()

# Keep the streaming query running
query.awaitTermination()