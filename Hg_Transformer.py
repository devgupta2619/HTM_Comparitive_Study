import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import time
import csv, datetime, math, os

# -------------------------------
# Transformer Model Functions
# -------------------------------

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(window_size, forecast_steps, model_dim, num_heads, ff_dim, num_transformer_blocks):
    inputs = layers.Input(shape=(window_size, 1))
    # Project input into higher dimension
    x = layers.Dense(model_dim)(inputs)
    # Add positional encoding
    pos_encoding = get_positional_encoding(window_size, model_dim)
    x = x + pos_encoding
    # Stack Transformer blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(model_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(forecast_steps)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def rmse(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))

def run_transformer_model(csv_file):
    # Load & Preprocess the Data
    data = pd.read_csv(csv_file, header=0)
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format="%m/%d/%y %H:%M", errors='coerce')
    data = data.dropna(subset=[data.columns[0]])
    data.sort_values(data.columns[0], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    consumption = data.iloc[:, 1].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    consumption_scaled = scaler.fit_transform(consumption)
    
    # Create sequences with a sliding window approach.
    window_size = 60
    forecast_steps = 5
    def create_sequences(data, window_size, forecast_steps):
        X, y = [], []
        for i in range(len(data) - window_size - forecast_steps + 1):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size : i + window_size + forecast_steps])
        return np.array(X), np.array(y)
    
    X, y_seq = create_sequences(consumption_scaled, window_size, forecast_steps)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    # Build and train the Transformer model.
    model_dim = 64
    num_heads = 4
    ff_dim = 128
    num_transformer_blocks = 2
    model = build_transformer_model(window_size, forecast_steps, model_dim, num_heads, ff_dim, num_transformer_blocks)
    model.summary()
    
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    start_time = time.time()
    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=20,
                        batch_size=32,
                        callbacks=[es],
                        verbose=1)
    training_time = time.time() - start_time
    
    loss, mae = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and true values.
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, forecast_steps)
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, forecast_steps)
    
    rmse_1 = rmse(y_test_inv[:, 0], predictions_inv[:, 0])
    rmse_5 = rmse(y_test_inv[:, -1], predictions_inv[:, -1])
    errors = np.abs(y_test_inv[:, 0] - predictions_inv[:, 0])
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    anomaly_likelihood = (errors - mean_error) / std_error
    
    result = {
        "model": "Transformer",
        "training_time": training_time,
        "test_loss": loss,
        "test_mae": mae,
        "rmse_1": rmse_1,
        "rmse_5": rmse_5,
        "anomaly_mean": mean_error,
        "anomaly_std": std_error,
        "history": history.history,
        "y_test_inv": y_test_inv,
        "predictions_inv": predictions_inv,
        "window_size": window_size,
        "forecast_steps": forecast_steps,
        "data": data,
        "scaler": scaler
    }
    return result

# -------------------------------
# HTM Model Functions (with Train-Test Split)
# -------------------------------

def run_htm_model(csv_file):
    from htm.bindings.sdr import SDR, Metrics
    from htm.encoders.rdse import RDSE, RDSE_Parameters
    from htm.encoders.date import DateEncoder
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
    from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
    from htm.bindings.algorithms import Predictor

    default_parameters = {
      'enc': {
          "value": {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
          "time": {'timeOfDay': (30, 1), 'weekend': 21}
      },
      'predictor': {'sdrc_alpha': 0.1},
      'sp': {'boostStrength': 3.0,
             'columnCount': 1638,
             'localAreaDensity': 0.04395604395604396,
             'potentialPct': 0.85,
             'synPermActiveInc': 0.04,
             'synPermConnected': 0.13999999999999999,
             'synPermInactiveDec': 0.006},
      'tm': {'activationThreshold': 17,
             'cellsPerColumn': 13,
             'initialPerm': 0.21,
             'maxSegmentsPerCell': 128,
             'maxSynapsesPerSegment': 64,
             'minThreshold': 10,
             'newSynapseCount': 32,
             'permanenceDec': 0.1,
             'permanenceInc': 0.1},
      'anomaly': {'period': 1000},
    }
    
    # Read CSV data.
    records = []
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin)
        headers = next(reader)
        next(reader)
        next(reader)
        for record in reader:
            records.append(record)
    
    # Define train-test split (80% training, 20% testing).
    split = int(0.8 * len(records))
    
    dateEncoder = DateEncoder(timeOfDay=default_parameters["enc"]["time"]["timeOfDay"],
                              weekend=default_parameters["enc"]["time"]["weekend"])
    scalarEncoderParams = RDSE_Parameters()
    scalarEncoderParams.size = default_parameters["enc"]["value"]["size"]
    scalarEncoderParams.sparsity = default_parameters["enc"]["value"]["sparsity"]
    scalarEncoderParams.resolution = default_parameters["enc"]["value"]["resolution"]
    scalarEncoder = RDSE(scalarEncoderParams)
    encodingWidth = dateEncoder.size + scalarEncoder.size
    enc_info = Metrics([encodingWidth], 999999999)
    
    spParams = default_parameters["sp"]
    sp = SpatialPooler(
        inputDimensions=(encodingWidth,),
        columnDimensions=(spParams["columnCount"],),
        potentialPct=spParams["potentialPct"],
        potentialRadius=encodingWidth,
        globalInhibition=True,
        localAreaDensity=spParams["localAreaDensity"],
        synPermInactiveDec=spParams["synPermInactiveDec"],
        synPermActiveInc=spParams["synPermActiveInc"],
        synPermConnected=spParams["synPermConnected"],
        boostStrength=spParams["boostStrength"],
        wrapAround=True
    )
    sp_info = Metrics(sp.getColumnDimensions(), 999999999)
    
    tmParams = default_parameters["tm"]
    tm = TemporalMemory(
        columnDimensions=(spParams["columnCount"],),
        cellsPerColumn=tmParams["cellsPerColumn"],
        activationThreshold=tmParams["activationThreshold"],
        initialPermanence=tmParams["initialPerm"],
        connectedPermanence=spParams["synPermConnected"],
        minThreshold=tmParams["minThreshold"],
        maxNewSynapseCount=tmParams["newSynapseCount"],
        permanenceIncrement=tmParams["permanenceInc"],
        permanenceDecrement=tmParams["permanenceDec"],
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
        maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"]
    )
    tm_info = Metrics([tm.numberOfCells()], 999999999)
    
    anomaly_history = AnomalyLikelihood(default_parameters["anomaly"]["period"])
    predictor = Predictor(steps=[1, 5], alpha=default_parameters["predictor"]['sdrc_alpha'])
    predictor_resolution = 1
    
    test_inputs = []
    predictions = {1: [], 5: []}
    anomaly_list = []
    anomalyProb_list = []
    
    start_time = time.time()
    for count, record in enumerate(records):
        dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
        consumption = float(record[1])
        
        if count >= split:
            test_inputs.append(consumption)
        
        dateBits = dateEncoder.encode(dateString)
        consumptionBits = scalarEncoder.encode(consumption)
        encoding = SDR(encodingWidth).concatenate([consumptionBits, dateBits])
        enc_info.addData(encoding)
        
        activeColumns = SDR(sp.getColumnDimensions())
        sp.compute(encoding, True, activeColumns)
        sp_info.addData(activeColumns)
        
        # Enable learning during training phase; disable during testing.
        if count < split:
            tm.compute(activeColumns, learn=True)
        else:
            tm.compute(activeColumns, learn=False)
        tm_info.addData(tm.getActiveCells().flatten())
        
        pdf = predictor.infer(tm.getActiveCells())
        for n in (1, 5):
            if pdf[n]:
                pred_val = np.argmax(pdf[n]) * predictor_resolution
            else:
                pred_val = float('nan')
            if count >= split:
                predictions[n].append(pred_val)
        
        if count >= split:
            anomaly_list.append(tm.anomaly)
            anomalyProb_list.append(anomaly_history.compute(tm.anomaly))
        
        if count < split:
            predictor.learn(count, tm.getActiveCells(), int(consumption / predictor_resolution))
    
    training_time = time.time() - start_time
    
    for n in predictions:
        for _ in range(n):
            predictions[n].insert(0, float('nan'))
            predictions[n].pop()
    
    accuracy = {1: 0, 5: 0}
    accuracy_samples = {1: 0, 5: 0}
    for idx, inp in enumerate(test_inputs):
        for n in predictions:
            val = predictions[n][idx]
            if not math.isnan(val):
                accuracy[n] += (inp - val) ** 2
                accuracy_samples[n] += 1
    for n in sorted(predictions):
        if accuracy_samples[n] > 0:
            accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** 0.5
        else:
            accuracy[n] = float('nan')
    
    anomaly_mean = np.mean(anomaly_list)
    anomaly_std = np.std(anomaly_list)
    
    test_inputs = np.array(test_inputs)
    predictions_1 = np.array(predictions[1])
    predictions_5 = np.array(predictions[5])
    anomaly_array = np.array(anomaly_list)
    anomalyProb_array = np.array(anomalyProb_list)
    
    result = {
        "model": "HTM",
        "training_time": training_time,
        "rmse_1": accuracy[1],
        "rmse_5": accuracy[5],
        "anomaly_mean": anomaly_mean,
        "anomaly_std": anomaly_std,
        "inputs": test_inputs,
        "predictions_1": predictions_1,
        "predictions_5": predictions_5,
        "anomaly": anomaly_array,
        "anomalyProb": anomalyProb_array
    }
    return result

# -------------------------------
# Comparative Plot & Summary Functions
# -------------------------------

def plot_comparative_results(transformer_results, htm_results):
    # Plot 1: Comparative Forecasts (1-Step Ahead)
    plt.figure(figsize=(12, 6))
    plt.title("Comparative Forecasts (1-Step Ahead)")
    transformer_actual = transformer_results["y_test_inv"][:, 0]
    transformer_pred = transformer_results["predictions_inv"][:, 0]
    plt.plot(transformer_actual, 'r-', label="Transformer Actual (Test)")
    plt.plot(transformer_pred, 'b--', label="Transformer Predicted (1-Step)")
    htm_actual = htm_results["inputs"]
    htm_pred = htm_results["predictions_1"]
    plt.plot(htm_actual, 'g-', label="HTM Actual (Test)")
    plt.plot(htm_pred, 'k--', label="HTM Predicted (1-Step)")
    plt.xlabel("Time Index")
    plt.ylabel("Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot 2: Comparative Anomaly Metrics
    plt.figure(figsize=(12, 6))
    plt.title("Comparative Anomaly Metrics")
    transformer_errors = np.abs(transformer_actual - transformer_pred)
    htm_anomaly = htm_results["anomaly"]
    plt.plot(transformer_errors, 'b-', label="Transformer Anomaly (Error)")
    plt.plot(htm_anomaly, 'r-', label="HTM Anomaly Score")
    plt.xlabel("Time Index")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot 3: Training Time Comparison
    plt.figure(figsize=(12, 6))
    plt.title("Training Time Comparison")
    models = ['Transformer', 'HTM']
    times = [transformer_results["training_time"], htm_results["training_time"]]
    plt.bar(models, times, color=['blue', 'green'])
    plt.ylabel("Time (seconds)")
    for i, t in enumerate(times):
        plt.text(i, t + 0.1, f"{t:.2f}s", ha='center', fontsize=12)
    plt.grid(axis='y')
    plt.show()
    
    # Plot 4: Transformer Training Loss over Epochs
    plt.figure(figsize=(12, 6))
    plt.title("Transformer Training Loss over Epochs")
    plt.plot(transformer_results["history"]["loss"], 'b-', label="Train Loss")
    if "val_loss" in transformer_results["history"]:
        plt.plot(transformer_results["history"]["val_loss"], 'r--', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def print_summary(transformer_results, htm_results):
    print("----- Comparative Metrics -----")
    print("Transformer Model:")
    print(f" Training Time: {transformer_results['training_time']:.2f} seconds")
    print(f" Test Loss (MSE): {transformer_results['test_loss']:.4f}")
    print(f" Test MAE: {transformer_results['test_mae']:.4f}")
    print(f" RMSE 1-step: {transformer_results['rmse_1']:.4f}")
    print(f" RMSE 5-step: {transformer_results['rmse_5']:.4f}")
    print(f" Anomaly Mean: {transformer_results['anomaly_mean']:.4f}")
    print(f" Anomaly Std: {transformer_results['anomaly_std']:.4f}")
    print("")
    print("HTM Model:")
    print(f" Training Time: {htm_results['training_time']:.2f} seconds")
    print(f" RMSE 1-step: {htm_results['rmse_1']:.4f}")
    print(f" RMSE 5-step: {htm_results['rmse_5']:.4f}")
    print(f" Anomaly Mean: {htm_results['anomaly_mean']:.4f}")
    print(f" Anomaly Std: {htm_results['anomaly_std']:.4f}")
    print("-------------------------------")

# -------------------------------
# Main Function to Run Both Models and Compare
# -------------------------------

def main():
    csv_file = "gymdata.csv"  # Ensure the CSV file is in the same directory
    
    print("Running Transformer Model...")
    transformer_results = run_transformer_model(csv_file)
    
    print("\nRunning HTM Model...")
    htm_results = run_htm_model(csv_file)
    
    print_summary(transformer_results, htm_results)
    plot_comparative_results(transformer_results, htm_results)

if __name__ == "__main__":
    main()
