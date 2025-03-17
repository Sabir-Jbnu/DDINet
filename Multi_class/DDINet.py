import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress Warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
#%% Set GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
#%% Define data paths
DATA_PATHS = {
    "train": "training_data/", # training data
    "test": "multi_class/s2/", # seen-unseen Test data
    "exp_test": "multi_class/s3/" # unseen-unseen test data
}

#%% Load Data
def load_data(path, file_name):
    full_path = os.path.join(path, file_name)
    data = np.load(full_path, allow_pickle=True)
    return data.reshape(data.shape[0], -1).astype("float32")  # Ensure float32 type

X_train = load_data(DATA_PATHS['train'], "concat_Morgan.npy")
y_train = np.load(os.path.join(DATA_PATHS['train'], "labels.npy"))
X_test = load_data(DATA_PATHS['test'], "concat_Morgan.npy")
y_test = np.load(os.path.join(DATA_PATHS['test'], "labels.npy"))
X_test1 = load_data(DATA_PATHS['exp_test'], "concat_Morgan.npy")
y_test1 = np.load(os.path.join(DATA_PATHS['exp_test'], "labels.npy"))

#%% Encode labels to categorical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_test1_encoded = label_encoder.transform(y_test1)

y_train_categorical = to_categorical(y_train_encoded, num_classes=106)
y_test_categorical = to_categorical(y_test_encoded, num_classes=106)
y_test1_categorical = to_categorical(y_test1_encoded, num_classes=106)

#%% Define the model
def my_model(input_shape, num_classes):
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
        Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy']
    )
    
    return model
#%% Train and validate using Stratified K-Fold Cross-Validation
def train_and_validate(X_train, y_train_encoded, y_train_categorical):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    history_by_fold = []
    validation_accuracies = []
    validation_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        print(f"Training fold {fold + 1}")
        model = my_model(input_shape=(X_train.shape[1],), num_classes=106)

        checkpoint_path = f"Morgan_weight_{fold + 1}.h5"
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path,
                            save_weights_only=False, save_best_only=True, verbose=1),
            EarlyStopping(monitor='loss', patience=15)
        ]

        # Use categorical labels for training the model
        history = model.fit(
            X_train[train_idx], y_train_categorical[train_idx],
            epochs=80, batch_size=256,
            validation_data=(X_train[val_idx], y_train_categorical[val_idx]),
            callbacks=callbacks
        )

        models.append(model)
        history_by_fold.append(history)
        validation_accuracies.append(history.history['val_accuracy'][-1])
        validation_losses.append(history.history['val_loss'][-1])

    mean_validation_accuracy = np.mean(validation_accuracies)
    mean_validation_loss = np.mean(validation_losses)
    print(f"Mean Validation Accuracy: {mean_validation_accuracy}")
    print(f"Mean Validation Loss: {mean_validation_loss}")

    return models, history_by_fold, validation_accuracies, validation_losses

# Train and validate the model
models, history_by_fold, validation_accuracies, validation_losses = train_and_validate(
    X_train, y_train_encoded, y_train_categorical)

#%% Plot training results
def plot_training_results(history_by_fold):
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

#%% Evaluate the final model on test datasets
final_model = models[-1]

# Evaluate on seen_unseen DDI dataset
final_loss, final_accuracy = final_model.evaluate(X_test, y_test_categorical)
print("Final Test Loss on seen_unseen DDI dataset:", final_loss)
print("Final Test Accuracy on seen_unseen DDI dataset:", final_accuracy)

# Evaluate on Unseen_unseen DDI dataset
final_loss1, final_accuracy1 = final_model.evaluate(X_test1, y_test1_categorical)
print("Final Test Loss on Unseen_unseen DDI dataset:", final_loss1)
print("Final Test Accuracy on Unseen_unseen DDI dataset:", final_accuracy1)
#%% Detailed metrics computation
def compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tn = np.diag(cm)
    fp = cm.sum(axis=0) - tn
    fn = cm.sum(axis=1) - tn
    tp = cm.sum() - (fp + fn + tn)

    epsilon = 1e-10  # Prevent division by zero
    specificity = np.nanmean(np.divide(tn, tn + fp + epsilon))
    sensitivity = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)

    # Binarize y_true for multi-class metrics
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # Filter out classes not present in y_true
    valid_classes = np.unique(y_true)
    y_pred_prob = y_pred_prob[:, valid_classes]
    y_true_bin = y_true_bin[:, valid_classes]

    # Handle ROC AUC calculation
    try:
        auroc = roc_auc_score(y_true_bin, y_pred_prob, average="weighted", multi_class="ovr")
    except ValueError:
        print("ROC AUC cannot be computed due to missing classes.")
        auroc = None

    # Calculate AUPR
    aupr_list = []
    for i, cls in enumerate(valid_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin[:, i], y_pred_prob[:, i])
        aupr_list.append(auc(recall_vals, precision_vals))
    aupr = np.mean(aupr_list) if aupr_list else None

    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1_score": f1,
        "MCC": mcc,
        "AUPR": aupr,
        "AUROC": auroc
    }

# Predict and evaluate for the seen-unseen test dataset
y_pred_prob = final_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

metrics = compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes=106)

# Print the metrics
print("Detailed Metrics for the seen-unseen Test Dataset:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Predict and evaluate for the unseen-unseen test dataset
y_pred_prob = final_model.predict(X_test1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test1, axis=1) if len(y_test1.shape) > 1 else y_test1

metrics = compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes=106)

# Print the metrics
print("Detailed Metrics for the unseen-unseen Test Dataset:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
#%%
