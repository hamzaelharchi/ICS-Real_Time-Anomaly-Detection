import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score

def process_data(df, scaler):
    """Preprocess the dataframe by scaling its numeric values."""
    df = df.applymap(lambda x: str(x).replace(",", ".")).astype(float)
    scaled_values = scaler.transform(df)
    return pd.DataFrame(scaled_values, columns=df.columns)

def create_windows(data, window_size):
    """Create sliding windows from the input data."""
    indices = np.arange(window_size)[None, :] + np.arange(data.shape[0] - window_size)[:, None]
    return data.values[indices]

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

        # Calculate anomaly scores
        score = (alpha * np.mean((batch - w1) ** 2, axis=1) + 
                 beta * np.mean((batch - w2) ** 2, axis=1))
        
        #print(score)
        results.append(score)

    return np.concatenate(results)

def load_scaler(filepath):
    """Load the MinMaxScaler from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_thresholds(filepath):
    """Load thresholds from a JSON file."""
    with open(filepath, "r") as f:
        return eval(json.load(f))

def main():
    """Main execution function."""
    # Configuration
    WINDOW_SIZE = 12

    # Load ONNX sessions, scaler, and thresholds
    encoder_session = ort.InferenceSession("notebooks/encoder.onnx")
    decoder1_session = ort.InferenceSession("notebooks/decoder1.onnx")
    decoder2_session = ort.InferenceSession("notebooks/decoder2.onnx")
    min_max_scaler = load_scaler("notebooks/scaler.pkl")
    thresholds = load_thresholds("notebooks/thresholds.json")

    # Load and preprocess test data
    attack_data = pd.read_csv("data/SWaT_Dataset_Attack_v0.csv")
    labels = (attack_data["Normal/Attack"] != 'Normal').astype(float).values
    attack_data = attack_data.drop(["Timestamp", "Normal/Attack"], axis=1)

    attack_data = process_data(attack_data, min_max_scaler)
    windows_attack = create_windows(attack_data, WINDOW_SIZE)

    # Flatten windows for testing
    test_data = windows_attack.reshape(len(windows_attack), -1)

    print("Testing model...")
    y_pred = test_model(encoder_session, decoder1_session, decoder2_session, test_data)
    print(y_pred)
    # Generate window-level labels
    window_labels = [labels[i:i + WINDOW_SIZE].max() for i in range(len(labels) - WINDOW_SIZE)]

    # Classify predictions based on threshold
    y_pred_labels = (y_pred > thresholds["threshold_youden"]).astype(float)

    # Evaluate and print metrics
    precision = precision_score(window_labels, y_pred_labels, pos_label=1)
    recall = recall_score(window_labels, y_pred_labels, pos_label=1)
    f1 = f1_score(window_labels, y_pred_labels, pos_label=1)

    print("\nModel Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()