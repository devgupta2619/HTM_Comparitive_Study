import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.metrics import precision_recall_fscore_support, classification_report

# --- HTM (NuPIC) Imports ---
from htm.bindings.algorithms import SpatialPooler, Classifier
from htm.bindings.sdr import SDR, Metrics

# --- CNN (Keras) Imports ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set Seaborn style for all plots
sns.set(style="whitegrid", palette="muted")


# ----------------------------
# Utility functions for Data Loading & Encoding
# ----------------------------
def load_ds(name, num_test, shape=None):
    """
    Fetch dataset from OpenML and split into train and test sets.
    
    :param name: dataset name on OpenML (e.g., 'mnist_784')
    :param num_test: number of samples to reserve for testing
    :param shape: new shape for each data point (e.g., [28, 28])
    :return: (train_labels, train_images, test_labels, test_images)
    """
    data = fetch_openml(name, version=1)
    sz = data['target'].shape[0]
    X = data['data']
    if shape is not None:
        X = np.reshape(X.to_numpy(), (sz, *shape))
    y = data['target'].astype(np.int32)
    train_labels = np.array(y[:sz - num_test])
    train_images = np.array(X[:sz - num_test])
    test_labels  = np.array(y[sz - num_test:])
    test_images  = np.array(X[sz - num_test:])
    return train_labels, train_images, test_labels, test_images

def encode(data, out):
    """
    Convert the grayscale image into a binary (black/white) SDR by thresholding
    at the image's mean.
    
    :param data: a single image (numpy array)
    :param out: an SDR object whose 'dense' attribute will hold the encoded data
    """
    out.dense = data >= np.mean(data)


# ----------------------------
# HTM Model (NuPIC) Function
# ----------------------------
def run_htm():
    """
    Train and test the HTM classifier on MNIST using a shared train-test split.
    
    Returns a dictionary with:
      - training_accuracy_curve: list of training accuracy (%) per iteration
      - final_accuracy: final test accuracy (%)
      - training_time: time taken for training (seconds)
      - metrics: a dict containing final precision, recall, and F1 score (in %)
      - test_data: list of (image, label) pairs from the HTM test set
      - model_components: dict containing encoder, Spatial Pooler, and classifier
    """
    print("Running HTM Model...")
    # Load MNIST from OpenML (using 10k test samples)
    train_labels, train_images, test_labels, test_images = load_ds('mnist_784', 10000, shape=[28, 28])
    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))
    random.shuffle(training_data)
    
    # HTM parameters (from original code)
    parameters = {
        'potentialRadius': 7,
        'boostStrength': 7.0,
        'columnDimensions': (79, 79),
        'dutyCyclePeriod': 1402,
        'localAreaDensity': 0.1,
        'minPctOverlapDutyCycle': 0.2,
        'potentialPct': 0.1,
        'stimulusThreshold': 6,
        'synPermActiveInc': 0.14,
        'synPermConnected': 0.5,
        'synPermInactiveDec': 0.02
    }
    
    # Initialize encoder and spatial pooler
    enc = SDR(train_images[0].shape)
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
        seed=0,
        spVerbosity=99,
        wrapAround=False
    )
    
    columns = SDR(sp.getColumnDimensions())
    _ = Metrics(columns, 99999999)  # For statistics (unused in plots)
    sdrc = Classifier()
    
    # Training loop: update classifier and record running accuracy
    accuracy_per_iteration = []
    correct_predictions = 0
    htm_start_time = time.time()
    for i in range(len(training_data)):
        img, lbl = training_data[i]
        encode(img, enc)
        sp.compute(enc, True, columns)
        sdrc.learn(columns, lbl)
        predicted_label = np.argmax(sdrc.infer(columns))
        if lbl == predicted_label:
            correct_predictions += 1
        accuracy = (correct_predictions / (i + 1)) * 100
        accuracy_per_iteration.append(accuracy)
    htm_end_time = time.time()
    htm_time = htm_end_time - htm_start_time
    
    # Testing loop: record predictions for evaluation metrics
    score = 0
    htm_true = []
    htm_pred = []
    for img, lbl in test_data:
        columns = SDR(sp.getColumnDimensions())
        encode(img, enc)
        sp.compute(enc, False, columns)
        pred = np.argmax(sdrc.infer(columns))
        htm_true.append(lbl)
        htm_pred.append(pred)
        if lbl == pred:
            score += 1
    final_accuracy = (score / len(test_data)) * 100

    # Compute precision, recall, and F1 score (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(htm_true, htm_pred, average='macro')
    htm_metrics = {
        'accuracy': final_accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    print('HTM Final Accuracy: {:.2f}%'.format(final_accuracy))
    print("HTM Classification Report:\n", classification_report(htm_true, htm_pred))
    
    # Plot training accuracy over iterations for HTM using Seaborn
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=list(range(len(accuracy_per_iteration))),
                 y=accuracy_per_iteration, marker="o", label="HTM Training Accuracy")
    plt.xlabel("Training Iterations", fontsize=24)
    plt.ylabel("Accuracy (%)", fontsize=24)
    plt.title("HTM Accuracy Over Training Iterations", fontsize=26)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
    
    return {'training_accuracy_curve': accuracy_per_iteration,
            'final_accuracy': final_accuracy,
            'training_time': htm_time,
            'metrics': htm_metrics,
            'test_data': test_data,
            'model_components': {'enc': enc, 'sp': sp, 'sdrc': sdrc}}


# ----------------------------
# CNN Model (Keras) Function with Consistent Train-Test Split
# ----------------------------
def run_cnn():
    """
    Train and test a simple CNN on MNIST using the same train-test split as HTM.
    
    Returns a dictionary with:
      - history: training history (accuracy and loss over epochs)
      - final_accuracy: final test accuracy (%)
      - training_time: time taken for training (seconds)
      - model: the trained Keras model
      - x_test, y_test: test set data
      - metrics: a dict containing final precision, recall, and F1 score (in %)
    """
    print("Running CNN Model...")
    # Use the same load_ds function to get a consistent train-test split from OpenML MNIST
    train_labels, train_images, test_labels, test_images = load_ds('mnist_784', 10000, shape=[28, 28])
    
    # Preprocess: convert to float32, normalize to [0,1], and reshape to (28,28,1)
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    # One-hot encode labels
    num_classes = 10
    y_train_cat = to_categorical(train_labels, num_classes)
    y_test_cat = to_categorical(test_labels, num_classes)
    
    # Build a simple CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    cnn_start_time = time.time()
    history = model.fit(train_images, y_train_cat, batch_size=128, epochs=5,
                        verbose=1, validation_split=0.1)
    cnn_end_time = time.time()
    cnn_time = cnn_end_time - cnn_start_time
    
    test_loss, test_acc = model.evaluate(test_images, y_test_cat, verbose=0)
    final_accuracy = test_acc * 100
    print('CNN Final Accuracy: {:.2f}%'.format(final_accuracy))
    
    # Plot CNN training & validation accuracy over epochs using Seaborn
    epochs = list(range(1, len(history.history['accuracy']) + 1))
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=history.history['accuracy'], marker="o", label="CNN Training Accuracy")
    sns.lineplot(x=epochs, y=history.history['val_accuracy'], marker="o", label="CNN Validation Accuracy")
    plt.xlabel("Epochs", fontsize=26)
    plt.ylabel("Accuracy", fontsize=26)
    plt.title("CNN Accuracy Over Epochs", fontsize=28)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
    
    # Compute precision, recall, and F1 score on test data using model predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels, average='macro')
    cnn_metrics = {
        'accuracy': final_accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
    print("CNN Classification Report:\n", classification_report(test_labels, predicted_labels))
    
    return {'history': history.history,
            'final_accuracy': final_accuracy,
            'training_time': cnn_time,
            'model': model,
            'x_test': test_images,
            'y_test': test_labels,
            'metrics': cnn_metrics}


# ----------------------------
# Comparative Analysis Functions
# ----------------------------
def comparative_analysis(htm_result, cnn_result):
    """
    Plot graphs to compare the two models including final metrics and training times.
    """
    # Compare final test accuracy and training time using Seaborn bar plots.
    models = ['HTM', 'CNN']
    final_accuracies = [htm_result['final_accuracy'], cnn_result['final_accuracy']]
    training_times = [htm_result['training_time'], cnn_result['training_time']]
    
    # Final Test Accuracy Comparison
    df_accuracy = pd.DataFrame({
        "Model": models,
        "Final Accuracy (%)": final_accuracies
    })
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(x="Model", y="Final Accuracy (%)", data=df_accuracy, palette=["blue", "green"])
    plt.ylim(0, 100)
    plt.title("Final Test Accuracy Comparison", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.2f}%', 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=22, color='black',
                     xytext=(0, 5), textcoords='offset points')
    plt.show()
    
    # Training Time Comparison
    df_time = pd.DataFrame({
        "Model": models,
        "Training Time (s)": training_times
    })
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(x="Model", y="Training Time (s)", data=df_time, palette=["blue", "green"])
    plt.title("Training Time Comparison", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}s', 
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=24, color='black',
                     xytext=(0, 5), textcoords='offset points')
    plt.show()
    
    # Plot Precision, Recall, and F1 Score Comparison as a grouped bar chart using Seaborn
    plot_metric_comparison(htm_result['metrics'], cnn_result['metrics'])


def plot_metric_comparison(htm_metrics, cnn_metrics):
    """
    Create a grouped bar chart comparing Accuracy, Precision, Recall, and F1 Score.
    """
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    htm_vals = [htm_metrics['accuracy'], htm_metrics['precision'], htm_metrics['recall'], htm_metrics['f1']]
    cnn_vals = [cnn_metrics['accuracy'], cnn_metrics['precision'], cnn_metrics['recall'], cnn_metrics['f1']]
    
    df_metrics = pd.DataFrame({
        "Metric": metric_names * 2,
        "Value": htm_vals + cnn_vals,
        "Model": ["HTM"] * len(metric_names) + ["CNN"] * len(metric_names)
    })
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Metric", y="Value", hue="Model", data=df_metrics, palette=["blue", "green"])
    plt.ylim(0, 100)
    plt.title("Comparison of Evaluation Metrics", fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=24, color='black',
                    xytext=(0, 5), textcoords='offset points')
    plt.show()


# ----------------------------
# Display Example Predictions
# ----------------------------
def display_example_predictions(htm_components, htm_test_data, cnn_model, cnn_x_test, cnn_y_test, num_examples=6):
    """
    Select a number of example images and display predictions from both HTM and CNN models.
    The function creates a figure with 2 rows: the top row shows HTM predictions, and
    the bottom row shows CNN predictions.
    """
    # Ensure we don't ask for more examples than available
    num_examples = min(num_examples, len(htm_test_data), len(cnn_x_test))
    
    # Randomly select indices from the HTM and CNN test sets
    htm_indices = random.sample(range(len(htm_test_data)), num_examples)
    cnn_indices = random.sample(range(len(cnn_x_test)), num_examples)
    
    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 3, 6))
    
    # HTM Predictions (top row)
    for i, idx in enumerate(htm_indices):
        img, actual = htm_test_data[idx]
        # Create a new SDR for columns for each prediction
        columns = SDR(htm_components['sp'].getColumnDimensions())
        encode(img, htm_components['enc'])
        htm_components['sp'].compute(htm_components['enc'], False, columns)
        predicted = np.argmax(htm_components['sdrc'].infer(columns))
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"HTM\nActual: {actual}\nPred: {predicted}", fontsize=18)
        axes[0, i].axis('off')
    
    # CNN Predictions (bottom row)
    for i, idx in enumerate(cnn_indices):
        img = cnn_x_test[idx]
        actual = cnn_y_test[idx]
        # CNN model expects (28,28,1) input; predict using a batch of 1 image.
        pred_probs = cnn_model.predict(np.expand_dims(img, axis=0))
        predicted = np.argmax(pred_probs)
        # Convert image to 2D for display
        display_img = np.squeeze(img)
        axes[1, i].imshow(display_img, cmap='gray')
        axes[1, i].set_title(f"CNN\nActual: {actual}\nPred: {predicted}", fontsize=18)
        axes[1, i].axis('off')
    
    plt.suptitle("Example Predictions from HTM (Top) and CNN (Bottom)", fontsize=26)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ----------------------------
# Main Function
# ----------------------------
def main():
    # Run the HTM model
    htm_result = run_htm()
    
    # Run the CNN model
    cnn_result = run_cnn()
    
    # Comparative graphs and summary
    comparative_analysis(htm_result, cnn_result)
    
    print("\nComparative Analysis Summary:")
    print("HTM Final Accuracy: {:.2f}%, Training Time: {:.2f} seconds".format(
        htm_result['final_accuracy'], htm_result['training_time']))
    print("HTM Metrics: Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(
        htm_result['metrics']['precision'], htm_result['metrics']['recall'], htm_result['metrics']['f1']))
    print("CNN Final Accuracy: {:.2f}%, Training Time: {:.2f} seconds".format(
        cnn_result['final_accuracy'], cnn_result['training_time']))
    print("CNN Metrics: Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(
        cnn_result['metrics']['precision'], cnn_result['metrics']['recall'], cnn_result['metrics']['f1']))
    
    # Display example predictions for both models
    display_example_predictions(htm_result['model_components'],
                                htm_result['test_data'],
                                cnn_result['model'],
                                cnn_result['x_test'],
                                cnn_result['y_test'],
                                num_examples=6)


if __name__ == '__main__':
    main()
