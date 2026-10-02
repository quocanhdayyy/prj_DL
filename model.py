# ============================================================
# DDI DEEP LEARNING PROJECT
# Step 1: Load data + Build model + Check architecture
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dropout,
    BatchNormalization,
    MaxPooling1D,
    Dense,
    LSTM,
    Bidirectional,
    Activation
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_SHAPE = (4096, 1)
NUM_CLASSES = 86
TEST_MODE = False
EPOCHS = 50
BATCH_SIZE = 32
BEST_MODEL_PATH = "best_model.keras"
HISTORY_PLOT_PATH = "training_history.png"

print("=" * 60)
print("DDI DEEP LEARNING PROJECT")
print("=" * 60)
print("TensorFlow version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data_from_npz(npz_path, num_classes=86, max_samples=None):
    print(f"\nLoading data from: {npz_path}")

    data = np.load(npz_path)

    X = data["X"]
    y = data["y"]

    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    # Conv1D requires:
    # (samples, sequence_length, channels)
    #
    # Original:
    # (N, 4096)
    #
    # After reshape:
    # (N, 4096, 1)
    X = np.expand_dims(X, axis=-1)

    # Convert labels:
    # 0, 1, ..., 85
    #
    # into one-hot vectors
    y = to_categorical(y, num_classes=num_classes)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


def load_train_validation(test_mode=False):
    """Load the train and validation datasets."""
    sample_limits = (1000, 300) if test_mode else (None, None)
    X_train, y_train = load_data_from_npz(
        "train_data.npz", NUM_CLASSES, sample_limits[0]
    )
    X_val, y_val = load_data_from_npz(
        "valid_data.npz", NUM_CLASSES, sample_limits[1]
    )
    return X_train, y_train, X_val, y_val


def load_test_data(test_mode=False):
    """Load the test dataset after training is complete."""
    test_limit = 300 if test_mode else None
    return load_data_from_npz("test_data.npz", NUM_CLASSES, test_limit)


def calculate_class_weights(y_train):
    """Calculate balanced weights for the labels present in the training set."""
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.unique(y_train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train_labels
    )
    return {
        int(cls): float(weight)
        for cls, weight in zip(classes, weights)
    }

# ============================================================
# 3. BUILD CNN + BiLSTM MODEL
# ============================================================

def build_model(input_shape, num_classes):

    model = Sequential([

        Input(shape=input_shape),

        # -------------------------
        # CNN Block 1
        # -------------------------
        Conv1D(
            filters=16,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2,
            strides=2
        ),

        Dropout(0.1),

        # -------------------------
        # CNN Block 2
        # -------------------------
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(0.1),

        # -------------------------
        # CNN Block 3
        # -------------------------
        Conv1D(
            filters=48,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(0.2),

        # -------------------------
        # CNN Block 4
        # -------------------------
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        # -------------------------
        # BiLSTM
        # -------------------------
        Bidirectional(
            LSTM(
                128,
                return_sequences=True
            )
        ),

        Activation("relu"),

        Bidirectional(
            LSTM(96)
        ),

        # -------------------------
        # Fully Connected
        # -------------------------
        Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),

        # 86-class classification
        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model


def compile_model(model):
    """Compile the CNN + BiLSTM model with the existing optimizer settings."""
    optimizer = RMSprop(learning_rate=0.000173)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training_history(history, output_path):
    """Save training and validation loss and accuracy plots."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Training accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Training history saved to: {output_path}")


def print_test_metrics(model, X_test, y_test):
    """Evaluate the best model and print micro and macro classification metrics."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=1)

    micro_precision, micro_recall, micro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print("\n" + "=" * 60)
    print("BEST MODEL TEST RESULTS")
    print("=" * 60)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")


def train_and_evaluate():
    """Train with callbacks, evaluate the best checkpoint, and save history."""
    X_train, y_train, X_val, y_val = load_train_validation(TEST_MODE)

    if TEST_MODE:
        training_epochs = 1
        print("\nTEST_MODE enabled: using reduced datasets and 1 epoch.")
    else:
        training_epochs = EPOCHS

    class_weights = calculate_class_weights(y_train)
    print("\nClass weights calculated successfully.")

    model = compile_model(build_model(INPUT_SHAPE, NUM_CLASSES))

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=training_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    X_test, y_test = load_test_data(TEST_MODE)
    print_test_metrics(best_model, X_test, y_test)
    plot_training_history(history, HISTORY_PLOT_PATH)


if __name__ == "__main__":
    train_and_evaluate()