from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import time


def preprocess_bank(df):
    df['Experience'] = df['Experience'].abs()
    return df

def run_nn_experiments(X_train_dict, y_train, X_test_dict, y_test, learning_rates, epochs_list):
    all_histories = {method: [] for method in X_train_dict.keys()}
    f1_scores = {method: [] for method in X_train_dict.keys()}
    accuracy_values = {method: [] for method in X_train_dict.keys()}
    loss_values = {method: [] for method in X_train_dict.keys()}

    for method, X_train in X_train_dict.items():
        X_test = X_test_dict[method]
        for lr in learning_rates:
            for epochs in epochs_list:
                print(f'\nRunning experiment for {method} with learning rate={lr} and epochs={epochs}')
                
                start_time = time.time()  # Start timing
                
                model = Sequential([
                    Dense(16, input_dim=X_train.shape[1], activation='relu'),
                    Dense(8, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

                # Evaluate the model
                loss, accuracy = model.evaluate(X_test, y_test)
                print(f'Test Accuracy: {accuracy:.4f}')
                accuracy_values[method].append(accuracy)

                # Make predictions
                y_pred = (model.predict(X_test) > 0.5).astype("int32")

                # Evaluate predictions
                f1 = f1_score(y_test, y_pred)
                print(f'Prediction F1 Score: {f1:.4f}')
                f1_scores[method].append(f1)

                # Store history for plotting later
                all_histories[method].append((lr, epochs, history))
                loss_values[method].append(history.history['loss'][-1])

                # Print data for graphs
                print(f'\nData for accuracy graph ({method}, lr={lr}, epochs={epochs}):')
                print('Training Accuracy:', history.history['accuracy'])
                print('Validation Accuracy:', history.history['val_accuracy'])

                print(f'Data for loss graph ({method}, lr={lr}, epochs={epochs}):')
                print('Training Loss:', history.history['loss'])
                print('Validation Loss:', history.history['val_loss'])

                # Measure and print training time
                training_time = time.time() - start_time
                print(f'Training time for {method} with lr={lr} and epochs={epochs}: {training_time:.2f} seconds')

    # Plot all training and validation accuracy values
    plt.figure(figsize=(14, 7))
    for method in X_train_dict.keys():
        for lr, epochs, history in all_histories[method]:
            plt.plot(history.history['accuracy'], label=f'Train Acc ({method})')
            plt.plot(history.history['val_accuracy'], label=f'Val Acc ({method})')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot all training and validation loss values
    plt.figure(figsize=(14, 7))
    for method in X_train_dict.keys():
        for lr, epochs, history in all_histories[method]:
            plt.plot(history.history['loss'], label=f'Train Loss ({method})')
            plt.plot(history.history['val_loss'], label=f'Val Loss ({method})')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    # Plot F1 score values
    plt.figure(figsize=(14, 7))
    for method in X_train_dict.keys():
        plt.plot(epochs_list, f1_scores[method], label=f'F1 Score ({method})', marker='o')
    plt.title('F1 Score vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.show()

    # Plot accuracy values
    plt.figure(figsize=(14, 7))
    for method in X_train_dict.keys():
        plt.plot(epochs_list, accuracy_values[method], label=f'Accuracy ({method})', marker='o')
    plt.title('Test Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss values
    plt.figure(figsize=(14, 7))
    for method in X_train_dict.keys():
        plt.plot(epochs_list, loss_values[method], label=f'Loss ({method})', marker='o')
    plt.title('Final Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()




def NN(df):
    X = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])
    y = df['Personal Loan']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the range of learning rates and epochs for experiments
    learning_rates = [0.01]
    epochs_list = [50]

    # Dictionary to store reduced datasets for each method
    X_train_dict = {}
    X_test_dict = {}

    # Run experiments with different learning rates and epochs for each dimensionality reduction method
    for dim_reduction_method in ['NN', 'PCA', 'ICA', 'RP']:
        print(f"\nRunning Neural Network experiments with {dim_reduction_method}")

        if dim_reduction_method == 'NN':
            X_train_dict[dim_reduction_method] = X_train_scaled
            X_test_dict[dim_reduction_method] = X_test_scaled
        elif dim_reduction_method == 'PCA':
            X_train_reduced, X_test_reduced, explained_variance = apply_pca(X_train_scaled, X_test_scaled)
            X_train_dict[dim_reduction_method] = X_train_reduced
            X_test_dict[dim_reduction_method] = X_test_reduced
        elif dim_reduction_method == 'ICA':
            X_train_reduced, X_test_reduced, kurtosis = apply_ica(X_train_scaled, X_test_scaled)
            X_train_dict[dim_reduction_method] = X_train_reduced
            X_test_dict[dim_reduction_method] = X_test_reduced
        elif dim_reduction_method == 'RP':
            X_train_reduced, X_test_reduced, reconstruction_error = apply_rp(X_train_scaled, X_test_scaled)
            X_train_dict[dim_reduction_method] = X_train_reduced
            X_test_dict[dim_reduction_method] = X_test_reduced

    # Run neural network experiments
    run_nn_experiments(X_train_dict, y_train, X_test_dict, y_test, learning_rates, epochs_list)


def apply_pca(X_train, X_test):
    pca = PCA(n_components=X_train.shape[1], random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    return X_train_pca, X_test_pca, explained_variance


def apply_ica(X_train, X_test):
    ica = FastICA(n_components=X_train.shape[1], random_state=42)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    # Compute kurtosis directly from the transformed components
    kurtosis = np.mean(np.abs(ica.components_ ** 4), axis=1)
    return X_train_ica, X_test_ica, kurtosis


def apply_rp(X_train, X_test):
    rp = GaussianRandomProjection(n_components=X_train.shape[1], random_state=42)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    reconstruction_error = np.mean((X_train - np.dot(X_train_rp, rp.components_.T)) ** 2)
    return X_train_rp, X_test_rp, reconstruction_error


df = pd.read_csv('../data/bank.csv')

print(df.head())
df = preprocess_bank(df)
NN(df)
