import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import onnx
import onnxruntime as ort
import json
import pickle

# Device Configuration
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Model Architecture
class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        
    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w

class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
    
    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        return loss1, loss2

    def validation_step(self, batch, n):
        with torch.no_grad():
            z = self.encoder(batch)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
            loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(
            epoch, result['val_loss1'], result['val_loss2']))

# Training and Evaluation Functions
def evaluate(model, val_loader, n, device):
    outputs = [model.validation_step(to_device(batch, device), n) 
              for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def train_model(model, train_loader, val_loader, epochs, device, 
                opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + 
                         list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + 
                         list(model.decoder2.parameters()))
    
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)
            
            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
        result = evaluate(model, val_loader, epoch+1, device)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def test_model(model, test_loader, device, alpha=0.5, beta=0.5):
    results = []
    with torch.no_grad():
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w1 = model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2, axis=1) + 
                         beta*torch.mean((batch-w2)**2, axis=1))
    return results

# Data Processing Functions
def process_data(df, min_max_scaler=None, fit=False):
    # Transform all columns into float64
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).replace(",", "."))
    df = df.astype(float)
    
    # Normalize data
    if fit:
        x_scaled = min_max_scaler.fit_transform(df.values)
    else:
        x_scaled = min_max_scaler.transform(df.values)
    return pd.DataFrame(x_scaled)

def create_windows(data, window_size):
    return data.values[np.arange(window_size)[None, :] + 
                      np.arange(data.shape[0]-window_size)[:, None]]

# Visualization Functions
def plot_training_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.figure(figsize=(10, 6))
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve_with_best_threshold(y_test, y_pred):
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

    # Calculate Youden's J statistic
    youdens_j = tpr - fpr
    best_idx = np.argmax(youdens_j)
    best_threshold_youden = thresholds[best_idx]

    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = pr_thresholds[best_f1_idx]

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot(fpr, 1 - fpr, 'r:', label="Random Classifier")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='blue', label=f"Best J (Threshold={best_threshold_youden:.3f})")
    plt.legend(loc='lower right')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.grid()
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.scatter(recall[best_f1_idx], precision[best_f1_idx], color='green', 
                label=f"Best F1 (Threshold={best_threshold_f1:.3f})")
    plt.legend(loc='upper right')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()

    return {"threshold_youden": best_threshold_youden, "threshold_f1": best_threshold_f1}


# Model Export Function
def export_model_to_onnx(model, input_sample, output_path="model.onnx"):
    model.eval()
    torch.onnx.export(
        model,
        input_sample,
        output_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")

def main():
    # Configuration
    WINDOW_SIZE = 12
    BATCH_SIZE = 7919
    N_EPOCHS = 2
    HIDDEN_SIZE = 100
    
    # Set device
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    normal_data = pd.read_csv("data/SWaT_Dataset_Normal_v0.csv")
    normal_data = normal_data.drop(["Timestamp", "Normal/Attack"], axis=1)
    
    attack_data = pd.read_csv("data/SWaT_Dataset_Attack_v0.csv")
    labels = [float(label != 'Normal') for label in attack_data["Normal/Attack"].values]
    attack_data = attack_data.drop(["Timestamp", "Normal/Attack"], axis=1)
    
    # Normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    normal_data = process_data(normal_data, min_max_scaler, fit=True)
    attack_data = process_data(attack_data, min_max_scaler, fit=False)

    
    # Save the scaler
    print("Saving scaler...")
    with open("notebooks/scaler.pkl", "wb") as f:
        pickle.dump(min_max_scaler, f)
    
    # Create windows
    windows_normal = create_windows(normal_data, WINDOW_SIZE)
    windows_attack = create_windows(attack_data, WINDOW_SIZE)
    
    # Prepare data loaders
    w_size = windows_normal.shape[1] * windows_normal.shape[2]
    z_size = windows_normal.shape[1] * HIDDEN_SIZE
    
    # Split normal data into train and validation
    split_idx = int(np.floor(.8 * windows_normal.shape[0]))
    windows_normal_train = windows_normal[:split_idx]
    windows_normal_val = windows_normal[split_idx:]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(
                [windows_normal_train.shape[0], w_size])
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(
                [windows_normal_val.shape[0], w_size])
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(
                [windows_attack.shape[0], w_size])
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    # Initialize and train model
    print("Initializing model...")
    model = UsadModel(w_size, z_size)
    model = to_device(model, device)
    
    print("Training model...")
    history = train_model(model, train_loader, val_loader, N_EPOCHS, device)
    plot_training_history(history)
    
    # Save model
    print("Saving model...")
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder1': model.decoder1.state_dict(),
        'decoder2': model.decoder2.state_dict()
    }, "notebooks/model.pth")
    
    # Test model
    print("Testing model...")
    results = test_model(model, test_loader, device)
    
    # Prepare labels for evaluation
    windows_labels = []
    for i in range(len(labels) - WINDOW_SIZE):
        windows_labels.append(list(np.int_(labels[i:i + WINDOW_SIZE])))
    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
    
    # Get predictions
    y_pred = np.concatenate([
        torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
        results[-1].flatten().detach().cpu().numpy()
    ])

    thresholds = plot_roc_curve_with_best_threshold(y_test, y_pred)
    print(f"Thresholds: {thresholds}")
    
    # Save thresholds
    print("Saving thresholds...")
    with open("notebooks/thresholds.json", "w") as f:
        json.dump(str(thresholds), f)
    
    # Calculate metrics
    y_pred_label = [1.0 if (score > thresholds["threshold_f1"]) else 0 for score in y_pred]
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print('Precision:', precision_score(y_test, y_pred_label, pos_label=1))
    print('Recall:', recall_score(y_test, y_pred_label, pos_label=1))
    print('F1 Score:', f1_score(y_test, y_pred_label, pos_label=1))
    
    # Export model to ONNX
    print("\nExporting model to ONNX...")
    sample_input = torch.randn(1, w_size).to(device)
    # Export encoder

    torch.onnx.export(model.encoder, sample_input, "notebooks/encoder.onnx",
                    export_params=True, opset_version=10)

    # Get encoded output for decoder input
    encoded_output = model.encoder(sample_input)

    # Export decoder1
    torch.onnx.export(model.decoder1, encoded_output, "notebooks/decoder1.onnx",
                    export_params=True, opset_version=10)

    # Export decoder2
    torch.onnx.export(model.decoder2, encoded_output, "notebooks/decoder2.onnx",
                    export_params=True, opset_version=10)


if __name__ == "__main__":
    main()