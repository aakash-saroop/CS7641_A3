import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def preprocess_bank(df):
    df['Experience'] = df['Experience'].abs()
    return df

# Function to load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('../data/bank.csv')
    
    # Preprocess the data
    df = preprocess_bank(df)
    
    # Define categorical and numerical columns
    categorical_features = ['Family', 'Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    numerical_features = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']

    # Preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Drop unnecessary columns
    X = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])
    
    # Fit and transform data
    X_preprocessed = preprocessor.fit_transform(X)
    
    return df, X_preprocessed

# Function to perform clustering
def perform_clustering(X_preprocessed):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_preprocessed)
    
    # Expectation-Maximisation Clustering
    em = GaussianMixture(n_components=3, random_state=42)
    em_labels = em.fit_predict(X_preprocessed)
    
    return kmeans_labels, em_labels

# Function to add cluster labels to DataFrame
def add_cluster_labels(df, kmeans_labels, em_labels):
    df['KMeans_Cluster'] = kmeans_labels
    df['EM_Cluster'] = em_labels
    return df

# Function to run neural network experiments and capture metrics for each epoch
def run_nn_experiments(X_train, y_train, X_test, y_test, learning_rates, epochs_list, method_name):
    all_histories = []
    all_accuracy_values = []
    all_loss_values = []
    all_f1_scores = []

    for lr in learning_rates:
        for epochs in epochs_list:
            print(f'\nRunning experiment with learning rate={lr} and epochs={epochs} for method {method_name}')
            model = Sequential([
                Dense(16, input_dim=X_train.shape[1], activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
            
            # Capture metrics after each epoch
            accuracy_values = history.history['accuracy']
            loss_values = history.history['loss']
            
            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f'Test Accuracy: {accuracy:.4f}')
            
            # Make predictions
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            
            # Evaluate predictions
            f1 = f1_score(y_test, y_pred)
            print(f'Prediction F1 Score: {f1:.4f}')
            
            # Store metrics for plotting later
            all_histories.append((lr, epochs, history))
            all_accuracy_values.append(accuracy_values)
            all_loss_values.append(loss_values)
            all_f1_scores.append(f1)

    # Return all captured metrics
    return all_histories, all_accuracy_values, all_loss_values, all_f1_scores

# Function to plot aggregated results across epochs
def plot_results(all_histories_with_clusters, all_histories_no_clusters):
    # Extract metrics for each epoch
    accuracy_values_with_clusters = [history[2].history['accuracy'] for history in all_histories_with_clusters]
    loss_values_with_clusters = [history[2].history['loss'] for history in all_histories_with_clusters]
    f1_scores_with_clusters = []

    accuracy_values_no_clusters = [history[2].history['accuracy'] for history in all_histories_no_clusters]
    loss_values_no_clusters = [history[2].history['loss'] for history in all_histories_no_clusters]
    f1_scores_no_clusters = []

    # Plot accuracy values
    plt.figure(figsize=(14, 7))
    for acc_with_clusters, acc_no_clusters in zip(accuracy_values_with_clusters, accuracy_values_no_clusters):
        plt.plot(acc_with_clusters, label='With Clusters', marker='o')
        plt.plot(acc_no_clusters, label='Without Clusters', marker='o')
    plt.title('Accuracy Comparison Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss values
    plt.figure(figsize=(14, 7))
    for loss_with_clusters, loss_no_clusters in zip(loss_values_with_clusters, loss_values_no_clusters):
        plt.plot(loss_with_clusters, label='With Clusters', marker='o')
        plt.plot(loss_no_clusters, label='Without Clusters', marker='o')
    plt.title('Loss Comparison Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()





# Main function to incorporate clustering and run NN experiments
def incorporate_clusters_and_run_nn():
    # Step 1: Load and preprocess data
    df, X_preprocessed = load_and_preprocess_data()

    # Step 2: Perform clustering
    kmeans_labels, em_labels = perform_clustering(X_preprocessed)

    # Step 3: Add cluster labels to DataFrame
    df = add_cluster_labels(df, kmeans_labels, em_labels)

    # Step 4: Prepare data for neural network
    X_with_clusters = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])
    y = df['Personal Loan']
    X_train, X_test, y_train, y_test = train_test_split(X_with_clusters, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    learning_rates = [0.01]
    epochs_list = [50]

    # Step 5: Run neural network experiments with clustering features
    all_histories_with_clusters, _, _, _ = \
        run_nn_experiments(X_train_scaled, y_train, X_test_scaled, y_test, learning_rates, epochs_list, "With Clusters")

    # Step 6: Prepare data without clustering features
    X_no_clusters = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan', 'KMeans_Cluster', 'EM_Cluster'])
    X_train_no_clusters, X_test_no_clusters, _, _ = train_test_split(X_no_clusters, y, test_size=0.2, random_state=42)

    X_train_no_clusters_scaled = scaler.fit_transform(X_train_no_clusters)
    X_test_no_clusters_scaled = scaler.transform(X_test_no_clusters)

    # Step 7: Run neural network experiments without clustering features
    all_histories_no_clusters, _, _, _ = \
        run_nn_experiments(X_train_no_clusters_scaled, y_train, X_test_no_clusters_scaled, y_test, learning_rates, epochs_list, "Without Clusters")

    # Step 8: Plot aggregated results across epochs
    plot_results(all_histories_with_clusters, all_histories_no_clusters)

# Execute the main function
if __name__ == "__main__":
    incorporate_clusters_and_run_nn()
