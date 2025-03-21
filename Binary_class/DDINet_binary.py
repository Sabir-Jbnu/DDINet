
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, auc)
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
from sklearn.manifold import TSNE
from matplotlib import cm, colors as mcolors
import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress Warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Adjust GPU index as needed
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

#%% Data Loading

def load_data(path, file_name):
    """Load and preprocess the data."""
    full_path = os.path.join(path, file_name)
    data = np.load(full_path, allow_pickle=True)
    return data.reshape(data.shape[0], -1).astype('float32')

# Define data paths
DATA_PATHS = {
    "train": "train_data/", # training data Path
    "test": "s2/",          # seen-unseen test data path
    "exp_test": "s3/"       # unseen-unseen test data path
}

# Load training and test data
X_train = load_data(DATA_PATHS['train'], "concat_Morgan.npy")
y_train = np.load(os.path.join(DATA_PATHS['train'], "labels.npy"))
X_test = load_data(DATA_PATHS['test'], "concat_Morgan.npy")
y_test = np.load(os.path.join(DATA_PATHS['test'], "labels.npy"))
X_test1 = load_data(DATA_PATHS['exp_test'], "concat_Morgan.npy")
y_test1 = np.load(os.path.join(DATA_PATHS['exp_test'], "labels.npy"))

# Print shapes of the loaded data
print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
print(f"X_test1 shape: {X_test1.shape}, dtype: {X_test1.dtype}")

#%% Model Definition
def my_model(input_shape, num_classes=2):
    """Define and compile the neural network model."""
    model = Sequential([
        Input(shape=input_shape),  # Specify the input shape here as the first layer
        Dense(436, activation='relu'),
        Dense(256, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        Dropout(0.3326933041901533),
        Dense(128, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        BatchNormalization(),
        Dropout(0.241),
        Dense(128, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        Dropout(0.3326933041901533),
        Dense(100, activation='relu', kernel_regularizer=l2(0.003524291993138782)),
        BatchNormalization(),
        Dropout(0.241),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    
    model.compile(
        loss='binary_crossentropy',  # Correct loss function for binary classification
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    return model

#%% Training and Validation Using Stratified K-Fold Cross-Validation

def train_and_validate(X_train, y_train):
    """Train the model using Stratified K-Fold Cross-Validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    history_by_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Training fold {fold + 1}")
        model = my_model(input_shape=(X_train.shape[1],))

        checkpoint_path = f"fold_{fold + 1}.h5"
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15, verbose=1)
        ]

        history = model.fit(
            X_train[train_idx], y_train[train_idx],
            epochs=80, batch_size=256,
            validation_data=(X_train[val_idx], y_train[val_idx]),
            callbacks=callbacks,
            verbose=1
        )

        models.append(model)
        history_by_fold.append(history)

    return models, history_by_fold

# Train and validate the model
models, history_by_fold = train_and_validate(X_train, y_train)

#%% Plot Training Results

def plot_training_results(history_by_fold):
    """Plot the training and validation accuracy and loss for each fold."""
    for fold, history in enumerate(history_by_fold, start=1):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Fold {fold} Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} Training and Validation Loss')
        plt.legend()
        plt.show()

plot_training_results(history_by_fold)

#%% Metrics Computation

def compute_metrics(y_true, y_pred):
    """Compute various classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1_score': f1,
        'MCC': mcc,
        "Confusion Matrix": cm
    }

# Predict and compute metrics for the test datasets
y_pred_prob = models[0].predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Binary classification
metrics_test = compute_metrics(y_test, y_pred)

print("Metrics for seen_unseen DDI Test Dataset:")
for metric, value in metrics_test.items():
    print(f"{metric}: {value}")

# Predict and compute metrics for the second test dataset
y_pred_prob1 = models[0].predict(X_test1)
y_pred1 = (y_pred_prob1 > 0.5).astype(int)  # Binary classification
metrics_test1 = compute_metrics(y_test1, y_pred1)

print("Metrics for Unseen_unseen DDI Test Dataset:")
for metric, value in metrics_test1.items():
    print(f"{metric}: {value}")

#%% Compute AUROC and AUPRC
def compute_auroc_auprc(y_true, y_pred_prob):
    """Compute AUROC and AUPR for evaluation."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return roc_auc, pr_auc

# Compute AUROC and AUPRC for the first test dataset (seen_unseen)
auroc_test, aupr_test = compute_auroc_auprc(y_test, y_pred_prob)

# Compute AUROC and AUPRC for the second test dataset (unseen_unseen)
auroc_test1, aupr_test1 = compute_auroc_auprc(y_test1, y_pred_prob1)

# Print AUROC and AUPRC values

print(f"AUPR for seen_unseen DDI Test Dataset: {aupr_test}")
print(f"AUROC for seen_unseen DDI Test Dataset: {auroc_test}")

print(f"AUPR for unseen_unseen DDI Test Dataset: {aupr_test1}")
print(f"AUROC for unseen_unseen DDI Test Dataset: {auroc_test1}")

#%% Plot Confusion Matrix

def plot_confusion_matrix(y_true, y_pred, title, filename_prefix):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, cmap='afmhot_r', square=True, cbar=True,
                xticklabels=np.arange(2), yticklabels=np.arange(2),
                annot=True, annot_kws={"size": 10}, fmt=".2f", linewidths=0.5, linecolor='black')

    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f"{filename_prefix}morgans2_confusion_matrix.png", dpi=800)
    plt.show()

plot_confusion_matrix(y_test, y_pred, "Confusion Matrix Heatmap", "seen_unseen_DDI")
plot_confusion_matrix(y_test1, y_pred1, "Confusion Matrix Heatmap", "unseen_unseen_DDI")
#%%
from sklearn.metrics import roc_auc_score

def plot_combined_pr_roc_curves(y_true, y_pred_prob, y_true1, y_pred_prob1, filename):
    """Plot combined PR and ROC curves for both datasets in one graph."""
    
    # Precision-Recall Curves
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    precision1, recall1, _ = precision_recall_curve(y_true1, y_pred_prob1)
    pr_auc = auc(recall, precision)
    pr_auc1 = auc(recall1, precision1)

    # ROC Curves
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    fpr1, tpr1, _ = roc_curve(y_true1, y_pred_prob1)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    roc_auc1 = roc_auc_score(y_true1, y_pred_prob1)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot Precision-Recall curves (solid lines)
    plt.plot(recall, precision, label=f'Seen-Unseen DDI PR curve (AUC = {pr_auc:.2f})', color='darkorange', linewidth=3)
    plt.plot(recall1, precision1, label=f'Unseen-Unseen DDI PR curve (AUC = {pr_auc1:.2f})', color='blue', linewidth=3)

    # Plot ROC curves (dashed lines)
    plt.plot(fpr, tpr, label=f'ROC curve of Seen-Unseen DDI (AUC = {roc_auc:.2f})', color='red', linestyle='--', linewidth=3)
    plt.plot(fpr1, tpr1, label=f' ROC curve of Unseen-Unseen DDI (AUC = {roc_auc1:.2f})', color='green', linestyle='--', linewidth=3)

    # Add diagonal line for ROC (Random Classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=2.5)

    # Add axis labels and title
    plt.xlabel('False Positive Rate / Recall', fontsize=16)
    plt.ylabel('True Positive Rate / Precision', fontsize=16)
    plt.title('Visualization of AUPR and AUROC Curves', fontsize=18)

    # Show gridlines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Customize the legend
    plt.legend(loc="lower right", fontsize=14, prop={'size': 10})

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{filename}_morgans2_pr_roc.png", dpi=800)

    # Show the plot
    plt.show()

# Example call to the function with appropriate arguments (make sure you pass the correct y_true and y_pred_prob)
plot_combined_pr_roc_curves(y_test, y_pred_prob, y_test1, y_pred_prob1, "combined")
#%%
