import time
import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
import json  # For handling JSON serialization

# Initialize Kafka Admin Client
admin_client = KafkaAdminClient(bootstrap_servers="localhost:9092")
existing_topics = admin_client.list_topics()
print(f"Existing topics: {existing_topics}")

# Create a Kafka producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    key_serializer=lambda k: k.encode('utf-8'),
    value_serializer=lambda v: v.encode('utf-8')  # Serialize messages as UTF-8 encoded strings
)

TOPIC_NAME = 'swat'

# Check if topic exists, create if not
if TOPIC_NAME not in existing_topics:
    topic_list = [NewTopic(name=TOPIC_NAME, num_partitions=1, replication_factor=1)]
    admin_client.create_topics(new_topics=topic_list, validate_only=False)

# Read CSV into a pandas DataFrame
df = pd.read_csv('./data/SWaT_Dataset_Attack_v0.csv')  # Replace 'Attack.csv' with your actual CSV file path

# Loop through each row of the CSV and send it row by row
for index, row in df.iterrows():
    # Convert the row to a dictionary and serialize it as a JSON string
    key = f"row-{index}"  # Unique key for each row
    value = json.dumps(row.to_dict())  # Serialize the row as JSON string
    # Send the row to Kafka
    producer.send(TOPIC_NAME, key=key, value=value)
    print(f"Sent row key: {key}, row value: {value}")

    # Optional: Wait before sending the next row
    time.sleep(5)  # Adjust sleep time as needed

# Close the producer after sending all messages
producer.close()
