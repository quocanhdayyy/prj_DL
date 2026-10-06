# DDI Deep Learning Project

This project trains an 86-class drug-drug interaction classifier using a CNN + BiLSTM model.

## Requirements

- Python 3.11
- TensorFlow 2.21.0
- NumPy
- scikit-learn
- Matplotlib

The configured Conda environment is:

```text
D:\conda_envs\deep
```

## Data

The training pipeline reads these preprocessed files from the project directory:

- `train_data.npz`
- `valid_data.npz`
- `test_data.npz`

Each sample contains 4096 input features. Labels are converted to 86-class one-hot vectors before training.

## Running The Model

Run the script with the project environment:

```text
D:\conda_envs\deep\python.exe model.py
```

The current configuration has `TEST_MODE = False`, so it uses the full training and validation datasets, trains for up to 50 epochs, and uses a batch size of 32.

The test dataset is loaded only after training and the best checkpoint has been loaded. This avoids holding the test data in memory during training.

## Test Mode

For a quick pipeline check, set this value in `model.py`:

```python
TEST_MODE = True
```

Test mode uses:

- 1,000 training samples
- 300 validation samples
- 300 test samples
- 1 training epoch
- Batch size 32

It still calculates class weights, saves a checkpoint, evaluates the model, prints metrics, and generates the training plot.

## Training Features

- CNN + Bidirectional LSTM architecture
- Balanced class weights for the imbalanced dataset
- Best-model checkpoint saved as `best_model.keras`
- Early stopping based on validation loss with patience 5
- Restoration of the best model weights
- Accuracy, micro precision/recall/F1, and macro precision/recall/F1 evaluation
- Training and validation loss/accuracy plot saved as `training_history.png`

Full training is CPU-intensive on Windows. Batch size 32 is intentionally retained to limit memory use.
