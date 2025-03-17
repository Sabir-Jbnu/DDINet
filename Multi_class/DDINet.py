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
from sklearn.metrics import (confusion_matrix, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress Warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
#%%
# Set GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Modify based on available GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

#%% Data loading function
def load_data(path, file_name):
    full_path = os.path.join(path, file_name)
    data = np.load(full_path, allow_pickle=True)
    return data.reshape(data.shape[0], -1).astype("float32")  # Ensure float32 for memory efficiency

# Define the data path
data_path = "input_file path"

# Load the data
X_train = load_data(data_path, "scaffold_train_morgan_fn.npy")
X_test = load_data(data_path, "scaffold_test_morgan_fn.npy")
#%%
# Inspect the first 5 rows of X_train
print("First 5 rows of X_train:")
print(X_train[:5])

# Inspect the first 5 rows of X_test
print("\nFirst 5 rows of X_test:")
print(X_test[:5])

# Separate features and labels
y_train = X_train[:, -1]  # Assuming the last column is 'labels'
X_train = X_train[:, :-1]  # All columns except the last one are features

y_test = X_test[:, -1]  # Assuming the last column is 'labels'
X_test = X_test[:, :-1]  # All columns except the last one are features
#%%
# Encode labels into binary variables
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y_categorical = to_categorical(y_encoded, num_classes=106)

y_test_encoded = label_encoder.transform(y_test)
y_test_categorical = to_categorical(y_test_encoded, num_classes=106)

#%% Print dataset shapes
print("Training data shape (X_train):", X_train.shape)
print("Training labels shape (y_train):", y_categorical.shape)
print("Test data shape (X_test):", X_test.shape)
print("Test labels shape (y_test):", y_test_categorical.shape)

#%% Model definition
def my_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
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
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

#%% Train and validate the model with Stratified K-Fold
def train_and_validate(X_train, y_train_categorical):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    history_by_fold = []
    validation_accuracies = []
    validation_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, np.argmax(y_train_categorical, axis=1))):
        print(f"Training fold {fold + 1}")
        model = my_model(input_shape=(X_train.shape[1],), num_classes=106)

        checkpoint_path = f"morgan_Scafold_{fold + 1}.h5"
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15)
        ]

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

    return models, history_by_fold

# Train the model
models, history_by_fold = train_and_validate(X_train, y_categorical)
#%%
# Final model evaluation
final_model = models[-1]  
final_loss, final_accuracy = final_model.evaluate(X_test, y_test_categorical)
print("Final Test Loss:", final_loss)
print("Final Test Accuracy:", final_accuracy)

#%% Compute detailed metrics
import numpy as np
def compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tn = np.diag(cm)
    fp = cm.sum(axis=0) - tn
    fn = cm.sum(axis=1) - tn
    tp = cm.sum() - (fp + fn + tn)

    epsilon = 1e-10
    specificity = np.nanmean(np.divide(tn, tn + fp + epsilon))
    sensitivity = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    try:
        auroc = roc_auc_score(y_true_bin, y_pred_prob, average="weighted", multi_class="ovr")
    except ValueError:
        auroc = None

    aupr_list = []
    for i in np.unique(y_true):
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

# Predict and evaluate
y_pred_prob = final_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test_categorical, axis=1)

metrics = compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes=106)

# Print metrics
print("\n### Detailed Metrics of Multi-classification using scaffold splitting ###")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
#%%












#%%
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, precision_recall_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import umap  # For dimensionality reduction visualization
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress Warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Set GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Modify based on available GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
#%%
# Data loading function
def load_data(path, file_name):
    full_path = os.path.join(path, file_name)
    data = np.load(full_path, allow_pickle=True)
    return data.reshape(data.shape[0], -1).astype("float32")  # Ensure the data is float32


# Define the data path
data_path = "seen_seen DDI/"

# Load the data
X = load_data(data_path, "concat_Morgan.npy")  # Feature data (Avalon dataset)
y = np.load(os.path.join(data_path, "labels.npy"))  # Labels

# Encode labels into binary variables
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=106)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical,
                                                    test_size=0.1, stratify=y_encoded, random_state=42)

# Save the split data (X_test and y_test) for future use
np.save(os.path.join(data_path, "X_testDeep_morgan.npy"), X_test)
np.save(os.path.join(data_path, "y_testDeep_morgan.npy"), y_test)

# Print the shapes of the training and test datasets
print("Training data shape (X_train):", X_train.shape)
print("Training labels shape (y_train):", y_train.shape)
print("Test data shape (X_test):", X_test.shape)
print("Test labels shape (y_test):", y_test.shape)
#%%
# Model definition
def my_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
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
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
#%%
# Train and validate the model with Stratified K-Fold
def train_and_validate(X_train, y_train_categorical):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    history_by_fold = []
    validation_accuracies = []
    validation_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, np.argmax(y_train_categorical, axis=1))):
        print(f"Training fold {fold + 1}")
        model = my_model(input_shape=(X_train.shape[1],), num_classes=106)

        checkpoint_path = f"morganS1_Deep_{fold + 1}.h5"
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=15)
        ]

        # Fit model
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

# Ensure the function train_and_validate is called to initialize the models list
models, history_by_fold, validation_accuracies, validation_losses = train_and_validate(X_train, y_train)
#%%
# Plot training results (Optional)
def plot_training_results(history_by_fold):
    plt.figure(figsize=(10, 6))

    # Plot validation accuracy for each fold
    for fold, history in enumerate(history_by_fold):
        plt.plot(history.history['val_accuracy'], label=f'Fold {fold + 1}')

    plt.title('Validation Accuracy across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.show()

    # Plot validation loss for each fold
    plt.figure(figsize=(10, 6))
    for fold, history in enumerate(history_by_fold):
        plt.plot(history.history['val_loss'], label=f'Fold {fold + 1}')

    plt.title('Validation Loss across Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()

# Call the function to plot the results
plot_training_results(history_by_fold)
#%%
# Now you can safely access the models
final_model = models[-1]  # Access the last model from the models list
final_loss, final_accuracy = final_model.evaluate(X_test, y_test)
print("Final Test Loss on seen_seen DDI dataset:", final_loss)
print("Final Test Accuracy on seen_seen DDI dataset:", final_accuracy)
#%%
# Compute detailed metrics function
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

    # Binarize the true labels for multi-class
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # Filter for valid classes
    valid_classes = np.unique(y_true)
    y_pred_prob = y_pred_prob[:, valid_classes]
    y_true_bin = y_true_bin[:, valid_classes]

    try:
        auroc = roc_auc_score(y_true_bin, y_pred_prob, average="weighted", multi_class="ovr")
    except ValueError:
        print("ROC AUC cannot be computed due to missing classes.")
        auroc = None

    # Calculate average AUPR for all classes
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

# Predict and evaluate for the test dataset
y_pred_prob = final_model.predict(X_test)  # Shape should be (num_samples, num_classes)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert to class labels (1D array)
y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test  # Convert one-hot to labels
#%%
# Now call the function with proper parameters
metrics = compute_detailed_metrics(y_true, y_pred, y_pred_prob, num_classes=106)

# Print the metrics
print("Detailed Metrics for the Test Dataset:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
#%%








#%%
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