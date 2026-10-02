# Project Progress

## Current Goal
Upgrade the original ML DDI project to a Deep Learning project using CNN-BiLSTM.

## Completed
- [x] Set up Python 3.11 environment
- [x] Installed TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, RDKit
- [x] Checked dataset
- [x] 86 output classes
- [x] Added class weights for imbalanced dataset
- [x] CNN-BiLSTM model runs successfully
- [x] Added ModelCheckpoint
- [x] Added EarlyStopping
- [x] Added evaluation metrics
- [x] Added training history plot
- [x] Added TEST_MODE
- [x] TEST_MODE passed on local machine
- [x] TEST_MODE passed on Google Colab GPU (Tesla T4)
- [x] Test pipeline successfully saves best_model.keras
- [x] Test pipeline successfully generates training_history.png
- [x] Moved project to personal GitHub repository

## Current Configuration
- Model: CNN + BiLSTM
- Input shape: (4096, 1)
- Classes: 86
- Train samples: 122,778
- Validation samples: 30,695
- Test samples: 38,369
- Batch size: 32
- Max epochs: 50
- Class weights: Enabled
- EarlyStopping: Enabled
- ModelCheckpoint: Enabled
- TEST_MODE: False

## Next Step
Run full training on Google Colab GPU.

After training:
1. Evaluate the best model on the full test set.
2. Record Accuracy, Micro F1 and Macro F1.
3. Save best_model.keras.
4. Save training_history.png.
5. Analyze the results and improve the model if necessary.