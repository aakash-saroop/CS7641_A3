import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

def preprocess_bank(df):
    """Preprocess the bank dataframe."""
    df['Experience'] = df['Experience'].abs()
    return df

def find_optimal_clusters(X, max_k):
    """Find optimal number of clusters using KMeans and silhouette scores."""
    iters = range(2, max_k + 1)
    sse = []
    silhouette_scores = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    plt.figure(figsize=(10, 5))
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(iters, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()

    optimal_k = iters[np.argmax(silhouette_scores)]
    print(f'Optimal number of clusters (KMeans): {optimal_k}')

    return optimal_k

def find_optimal_pca_components(X):
    """Find optimal number of PCA components to explain 95% variance."""
    pca = PCA().fit(X)
    explained_variances = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance for Optimal Number of Components')
    plt.show()

    optimal_components = np.argmax(explained_variances >= 0.95) + 1
    print(f'Optimal number of PCA components: {optimal_components}')

    return optimal_components

def calculate_correlation(df, cluster_col, target_col):
    """Calculate and print correlation between cluster labels and target variable."""
    contingency_table = pd.crosstab(df[cluster_col], df[target_col])
    correlation = contingency_table.apply(lambda r: r / r.sum(), axis=1)
    print(f'\nCorrelation between {cluster_col} and {target_col}:')
    print(correlation)
    return correlation

def visualize_tsne(X, labels, title):
    """Visualize clusters using t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def bank_clustering():
    """Run clustering analysis on bank data."""
    df = pd.read_csv('../data/bank.csv')
    df = preprocess_bank(df)

    X = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])
    y = df['Personal Loan']

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

    X_preprocessed = preprocessor.fit_transform(X)

    # Find optimal clusters for KMeans
    optimal_kmeans_clusters = find_optimal_clusters(X_preprocessed, 10)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_kmeans_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_preprocessed)

    # Expectation-Maximisation Clustering
    em = GaussianMixture(n_components=optimal_kmeans_clusters, random_state=42)
    em_labels = em.fit_predict(X_preprocessed)

    # Find optimal PCA components
    optimal_pca_components = find_optimal_pca_components(X_preprocessed)

    # Add cluster labels to the dataframe
    df['KMeans_Cluster'] = kmeans_labels
    df['EM_Cluster'] = em_labels

    # Calculate correlation between clusters and target
    kmeans_correlation = calculate_correlation(df, 'KMeans_Cluster', 'Personal Loan')
    em_correlation = calculate_correlation(df, 'EM_Cluster', 'Personal Loan')

    # Visualize the clusters using t-SNE
    visualize_tsne(X_preprocessed, kmeans_labels, 'K-Means Clustering (t-SNE)')
    visualize_tsne(X_preprocessed, em_labels, 'Expectation-Maximisation Clustering (t-SNE)')

def select_random(df):
    sampled_df = df.sample(n=28000, random_state=42)
    return sampled_df

def credit_card_clustering():
    """Run clustering analysis on credit card data."""
    df = pd.read_csv('../data/credit.csv')
    df = select_random(df)

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal clusters for KMeans
    optimal_kmeans_clusters = find_optimal_clusters(X_scaled, 10)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_kmeans_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Expectation-Maximisation Clustering
    em = GaussianMixture(n_components=optimal_kmeans_clusters, random_state=42)
    em_labels = em.fit_predict(X_scaled)

    # Find optimal PCA components
    optimal_pca_components = find_optimal_pca_components(X_scaled)

    # Add cluster labels to the dataframe
    df['KMeans_Cluster'] = kmeans_labels
    df['EM_Cluster'] = em_labels

    # Calculate correlation between clusters and target
    kmeans_correlation = calculate_correlation(df, 'KMeans_Cluster', 'Class')
    em_correlation = calculate_correlation(df, 'EM_Cluster', 'Class')

    # Visualize the clusters using t-SNE
    visualize_tsne(X_scaled, kmeans_labels, 'K-Means Clustering (t-SNE)')
    visualize_tsne(X_scaled, em_labels, 'Expectation-Maximisation Clustering (t-SNE)')

# Execute clustering functions
bank_clustering()
credit_card_clustering()
