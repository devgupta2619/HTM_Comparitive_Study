#!/usr/bin/env python3
"""
Comparative NLP Analysis: HTM vs LSTM for Sentiment Classification on IMDB Reviews

- HTM Model:
  Uses a bag-of-words (binary vector) representation and NuPIC's Spatial Pooler +
  Classifier to predict sentiment.

- LSTM Model:
  Uses an Embedding layer and an LSTM to capture sequential context for sentiment classification.

Both models are evaluated using accuracy, precision, recall, and F1 score.
Comparative graphs are plotted with Seaborn and example reviews with predictions are displayed.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, classification_report

# --- HTM (NuPIC) Imports ---
from htm.bindings.algorithms import SpatialPooler, Classifier
from htm.bindings.sdr import SDR

# --- LSTM (Keras) Imports ---
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Set Seaborn style for plots
sns.set(style="whitegrid", palette="muted")

# ----------------------------
# Utility Functions for NLP HTM
# ----------------------------
VOCAB_SIZE = 1000  # Limit vocabulary to 1000 words

def sequences_to_bow(sequences, vocab_size=VOCAB_SIZE):
    """
    Convert a list of sequences (list of integers) to a bag-of-words binary matrix.
    Each review is converted to a binary vector of length 'vocab_size'.
    """
    num_samples = len(sequences)
    bow = np.zeros((num_samples, vocab_size), dtype=np.bool_)
    for i, seq in enumerate(sequences):
        for word in seq:
            if word < vocab_size:
                bow[i, word] = True
    return bow

# ----------------------------
# HTM Model for NLP
# ----------------------------
def run_htm_nlp():
    """
    Train and test an HTM classifier on the IMDB dataset.
    Uses bag-of-words binary encoding.
    
    Returns a dictionary with:
      - training_accuracy_curve: list of training accuracy (%) per iteration
      - final_accuracy: final test accuracy (%)
      - training_time: training duration (seconds)
      - metrics: dict with final precision, recall, and F1 (in %)
      - test_data: list of (bow_vector, label) pairs for test samples
      - model_components: dict with 'enc', 'sp', and 'sdrc'
    """
    print("Running HTM NLP Model (Bag-of-Words)...")
    # Load IMDB dataset (using Keras) with vocabulary limited to VOCAB_SIZE words.
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    
    # Convert sequences to bag-of-words binary vectors.
    x_train_bow = sequences_to_bow(x_train, VOCAB_SIZE)
    x_test_bow  = sequences_to_bow(x_test, VOCAB_SIZE)
    
    # For HTM, each input is a 1D vector of length VOCAB_SIZE.
    input_shape = (VOCAB_SIZE,)
    
    # Set HTM parameters (adjusted for 1D input)
    parameters = {
        'potentialRadius': VOCAB_SIZE,         # Use full range of input indices.
        'boostStrength': 7.0,
        'columnDimensions': (256,),            # 256 columns for 1D input.
        'dutyCyclePeriod': 1000,
        'localAreaDensity': 0.1,
        'minPctOverlapDutyCycle': 0.2,
        'potentialPct': 0.2,
        'stimulusThreshold': 2,
        'synPermActiveInc': 0.14,
        'synPermConnected': 0.5,
        'synPermInactiveDec': 0.02
    }
    
    # Initialize a dummy SDR encoder for 1D input.
    enc = SDR(input_shape)
    
    sp = SpatialPooler(
        inputDimensions=enc.dimensions,
        columnDimensions=parameters['columnDimensions'],
        potentialRadius=parameters['potentialRadius'],
        potentialPct=parameters['potentialPct'],
        globalInhibition=True,
        localAreaDensity=parameters['localAreaDensity'],
        stimulusThreshold=int(round(parameters['stimulusThreshold'])),
        synPermInactiveDec=parameters['synPermInactiveDec'],
        synPermActiveInc=parameters['synPermActiveInc'],
        synPermConnected=parameters['synPermConnected'],
        minPctOverlapDutyCycle=parameters['minPctOverlapDutyCycle'],
        dutyCyclePeriod=int(round(parameters['dutyCyclePeriod'])),
        boostStrength=parameters['boostStrength'],
        seed=42,
        spVerbosity=0,
        wrapAround=False
    )
    
    sdrc = Classifier()
    
    # Training loop for HTM.
    training_accuracy_curve = []
    correct = 0
    start_time = time.time()
    num_train = len(x_train_bow)
    for i in range(num_train):
        vec = x_train_bow[i]
        label = y_train[i]
        enc.dense = vec.copy()
        columns = SDR(sp.getColumnDimensions())
        sp.compute(enc, True, columns)
        sdrc.learn(columns, label)
        pred = np.argmax(sdrc.infer(columns))
        if pred == label:
            correct += 1
        training_accuracy_curve.append((correct / (i + 1)) * 100)
    training_time = time.time() - start_time
    
    # Testing loop for HTM.
    correct_test = 0
    htm_true = []
    htm_pred = []
    test_data = []  # store (bow_vector, label) pairs
    for i in range(len(x_test_bow)):
        vec = x_test_bow[i]
        label = y_test[i]
        test_data.append((vec, label))
        enc.dense = vec.copy()
        columns = SDR(sp.getColumnDimensions())
        sp.compute(enc, False, columns)
        pred = np.argmax(sdrc.infer(columns))
        htm_true.append(label)
        htm_pred.append(pred)
        if pred == label:
            correct_test += 1
    final_accuracy = (correct_test / len(x_test_bow)) * 100
    
    precision, recall, f1, _ = precision_recall_fscore_support(htm_true, htm_pred, average='macro')
    metrics = {
        'accuracy': final_accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    
    print("HTM NLP Final Accuracy: {:.2f}%".format(final_accuracy))
    print("HTM NLP Classification Report:\n", classification_report(htm_true, htm_pred))
    
    # Plot training accuracy for HTM.
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=list(range(len(training_accuracy_curve))),
                 y=training_accuracy_curve, marker="o", label="HTM Training Accuracy")
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("HTM NLP Accuracy Over Training Iterations")
    plt.legend()
    plt.show()
    
    return {'training_accuracy_curve': training_accuracy_curve,
            'final_accuracy': final_accuracy,
            'training_time': training_time,
            'metrics': metrics,
            'test_data': test_data,
            'model_components': {'enc': enc, 'sp': sp, 'sdrc': sdrc}}

# ----------------------------
# LSTM Model for NLP
# ----------------------------
def run_lstm_nlp():
    """
    Train and test an LSTM classifier on the IMDB dataset.
    Uses raw sequences with an Embedding layer and LSTM.
    
    Returns a dictionary with:
      - history: training history (accuracy, loss over epochs)
      - final_accuracy: final test accuracy (%)
      - training_time: training duration (seconds)
      - model: the trained Keras LSTM model
      - x_test: padded test sequences
      - y_test: test labels
      - metrics: dict with final precision, recall, and F1 (in %)
    """
    print("Running LSTM NLP Model...")
    maxlen = 500  # maximum review length
    embedding_dim = 50
    
    # Load IMDB dataset (vocabulary size VOCAB_SIZE)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    
    # Pad sequences to a fixed length.
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test  = pad_sequences(x_test, maxlen=maxlen)
    
    # Build LSTM model.
    model = Sequential([
        Embedding(VOCAB_SIZE, embedding_dim, input_length=maxlen),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=1)
    training_time = time.time() - start_time
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    final_accuracy = test_acc * 100
    
    predictions = (model.predict(x_test) > 0.5).astype("int32").flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    metrics = {
        'accuracy': final_accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    
    print("LSTM NLP Final Accuracy: {:.2f}%".format(final_accuracy))
    print("LSTM NLP Classification Report:\n", classification_report(y_test, predictions))
    
    # Plot LSTM training and validation accuracy.
    epochs = list(range(1, len(history.history['accuracy']) + 1))
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=history.history['accuracy'], marker="o", label="LSTM Training Accuracy")
    sns.lineplot(x=epochs, y=history.history['val_accuracy'], marker="o", label="LSTM Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("LSTM NLP Accuracy Over Epochs")
    plt.legend()
    plt.show()
    
    return {'history': history.history,
            'final_accuracy': final_accuracy,
            'training_time': training_time,
            'model': model,
            'x_test': x_test,
            'y_test': y_test,
            'metrics': metrics}

# ----------------------------
# Comparative Analysis Functions (NLP)
# ----------------------------
def comparative_analysis_nlp(htm_result, lstm_result):
    """
    Plot comparative graphs (final accuracy, training time, and evaluation metrics)
    for the HTM and LSTM NLP models.
    """
    models = ['HTM NLP', 'LSTM NLP']
    final_accuracies = [htm_result['final_accuracy'], lstm_result['final_accuracy']]
    training_times = [htm_result['training_time'], lstm_result['training_time']]
    
    # Final Accuracy Comparison.
    df_acc = pd.DataFrame({
        "Model": models,
        "Final Accuracy (%)": final_accuracies
    })
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(x="Model", y="Final Accuracy (%)", data=df_acc, palette=["blue", "green"])
    plt.ylim(0, 100)
    plt.title("Final Accuracy Comparison (NLP)")
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.2f}%', 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=12, color='black',
                     xytext=(0, 5), textcoords='offset points')
    plt.show()
    
    # Training Time Comparison.
    df_time = pd.DataFrame({
        "Model": models,
        "Training Time (s)": training_times
    })
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(x="Model", y="Training Time (s)", data=df_time, palette=["blue", "green"])
    plt.title("Training Time Comparison (NLP)")
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}s', 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=12, color='black',
                     xytext=(0, 5), textcoords='offset points')
    plt.show()
    
    # Metrics Comparison.
    plot_metric_comparison_nlp(htm_result['metrics'], lstm_result['metrics'])


def plot_metric_comparison_nlp(htm_metrics, lstm_metrics):
    """
    Create a grouped bar chart comparing evaluation metrics for NLP models.
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    htm_vals = [htm_metrics['accuracy'], htm_metrics['precision'], htm_metrics['recall'], htm_metrics['f1']]
    lstm_vals = [lstm_metrics['accuracy'], lstm_metrics['precision'], lstm_metrics['recall'], lstm_metrics['f1']]
    
    df_metrics = pd.DataFrame({
        "Metric": metrics_names * 2,
        "Value": htm_vals + lstm_vals,
        "Model": ["HTM NLP"] * len(metrics_names) + ["LSTM NLP"] * len(metrics_names)
    })
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Value", hue="Model", data=df_metrics, palette=["blue", "green"])
    plt.ylim(0, 100)
    plt.title("Evaluation Metrics Comparison (NLP)")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')
    plt.show()


# ----------------------------
# Display Example Predictions for NLP
# ----------------------------
def display_example_predictions_nlp(htm_components, htm_test_data, lstm_model, lstm_x_test, lstm_y_test, num_examples=6):
    """
    Display example reviews (truncated) along with predictions from both HTM and LSTM NLP models.
    Top row shows HTM predictions (using bag-of-words) and bottom row shows LSTM predictions.
    """
    num_examples = min(num_examples, len(htm_test_data), len(lstm_x_test))
    # Random indices for examples.
    htm_indices = random.sample(range(len(htm_test_data)), num_examples)
    lstm_indices = random.sample(range(len(lstm_x_test)), num_examples)
    
    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 4, 8))
    
    # HTM Predictions (top row).
    for i, idx in enumerate(htm_indices):
        vec, actual = htm_test_data[idx]
        columns = SDR(htm_components['sp'].getColumnDimensions())
        htm_components['enc'].dense = vec.copy()
        htm_components['sp'].compute(htm_components['enc'], False, columns)
        pred = np.argmax(htm_components['sdrc'].infer(columns))
        axes[0, i].text(0.5, 0.5, f"Actual: {actual}\nPred: {pred}",
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12, wrap=True)
        axes[0, i].set_title("HTM NLP")
        axes[0, i].axis('off')
    
    # LSTM Predictions (bottom row).
    for i, idx in enumerate(lstm_indices):
        review_seq = lstm_x_test[idx]
        actual = lstm_y_test[idx]
        pred_prob = lstm_model.predict(np.expand_dims(review_seq, axis=0))
        pred = int(pred_prob[0][0] > 0.5)
        axes[1, i].text(0.5, 0.5, f"Actual: {actual}\nPred: {pred}",
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12, wrap=True)
        axes[1, i].set_title("LSTM NLP")
        axes[1, i].axis('off')
    
    plt.suptitle("Example Predictions: HTM NLP (Top) vs LSTM NLP (Bottom)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------
# Main Function (NLP)
# ----------------------------
def main():
    # Run the HTM model.
    htm_result = run_htm_nlp()
    
    # Run the LSTM model.
    lstm_result = run_lstm_nlp()
    
    # Comparative analysis.
    comparative_analysis_nlp(htm_result, lstm_result)
    
    print("\nNLP Comparative Analysis Summary:")
    print("HTM NLP Final Accuracy: {:.2f}%, Training Time: {:.2f} seconds".format(
        htm_result['final_accuracy'], htm_result['training_time']))
    print("HTM NLP Metrics: Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(
        htm_result['metrics']['precision'], htm_result['metrics']['recall'], htm_result['metrics']['f1']))
    print("LSTM NLP Final Accuracy: {:.2f}%, Training Time: {:.2f} seconds".format(
        lstm_result['final_accuracy'], lstm_result['training_time']))
    print("LSTM NLP Metrics: Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(
        lstm_result['metrics']['precision'], lstm_result['metrics']['recall'], lstm_result['metrics']['f1']))
    
    # Display example predictions for NLP models.
    display_example_predictions_nlp(htm_result['model_components'],
                                    htm_result['test_data'],
                                    lstm_result['model'],
                                    lstm_result['x_test'],
                                    lstm_result['y_test'],
                                    num_examples=6)


if __name__ == '__main__':
    main()
