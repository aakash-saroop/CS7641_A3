import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from sklearn.metrics import adjusted_rand_score

def select_random(df):
    sampled_df = df.sample(n=28000, random_state=42)
    return sampled_df

def evaluate_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    return X_pca, explained_variance

def evaluate_ica(X, n_components):
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    kurt = kurtosis(X_ica, axis=0)
    return X_ica, kurt

def evaluate_rp(X, n_components):
    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X)
    # Ensure that pseudo_inverse has the correct shape
    projection_matrix = rp.components_.T
    pseudo_inverse = np.linalg.pinv(projection_matrix)
    X_reconstructed = np.dot(X_rp, pseudo_inverse)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    return X_rp, reconstruction_error


def credit_card_clustering():
    df = pd.read_csv('../data/credit.csv')
    df = select_random(df)

    X = df.drop(columns=['Class'])
    y = df['Class']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering in original space
    kmeans_original = KMeans(n_clusters=2, random_state=42)
    kmeans_labels_original = kmeans_original.fit_predict(X_scaled)

    # EM Clustering in original space
    em_original = GaussianMixture(n_components=2, random_state=42)
    em_labels_original = em_original.fit_predict(X_scaled)

    # Dimensionality Reduction
    pca_components = 9
    X_pca, explained_variance = evaluate_pca(X_scaled, pca_components)

    ica_components = 9
    X_ica, kurt = evaluate_ica(X_scaled, ica_components)

    rp_components = 9
    X_rp, reconstruction_error = evaluate_rp(X_scaled, rp_components)

    # K-Means Clustering in reduced spaces
    kmeans_pca = KMeans(n_clusters=2, random_state=42)
    kmeans_labels_pca = kmeans_pca.fit_predict(X_pca)

    kmeans_ica = KMeans(n_clusters=2, random_state=42)
    kmeans_labels_ica = kmeans_ica.fit_predict(X_ica)

    kmeans_rp = KMeans(n_clusters=2, random_state=42)
    kmeans_labels_rp = kmeans_rp.fit_predict(X_rp)

    # EM Clustering in reduced spaces
    em_pca = GaussianMixture(n_components=2, random_state=42)
    em_labels_pca = em_pca.fit_predict(X_pca)

    em_ica = GaussianMixture(n_components=2, random_state=42)
    em_labels_ica = em_ica.fit_predict(X_ica)

    em_rp = GaussianMixture(n_components=2, random_state=42)
    em_labels_rp = em_rp.fit_predict(X_rp)

    # Add cluster labels to the dataframe
    df['KMeans_Cluster_Original'] = kmeans_labels_original
    df['KMeans_Cluster_PCA'] = kmeans_labels_pca
    df['KMeans_Cluster_ICA'] = kmeans_labels_ica
    df['KMeans_Cluster_RP'] = kmeans_labels_rp

    df['EM_Cluster_Original'] = em_labels_original
    df['EM_Cluster_PCA'] = em_labels_pca
    df['EM_Cluster_ICA'] = em_labels_ica
    df['EM_Cluster_RP'] = em_labels_rp

    # Visualize the clusters using PCA for dimensionality reduction
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['KMeans_Cluster_Original'], palette='viridis')
    plt.title('K-Means Clustering (Original Space)')

    plt.subplot(2, 3, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['KMeans_Cluster_PCA'], palette='viridis')
    plt.title('K-Means Clustering (PCA)')

    plt.subplot(2, 3, 3)
    sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=df['KMeans_Cluster_ICA'], palette='viridis')
    plt.title('K-Means Clustering (ICA)')

    plt.subplot(2, 3, 4)
    sns.scatterplot(x=X_rp[:, 0], y=X_rp[:, 1], hue=df['KMeans_Cluster_RP'], palette='viridis')
    plt.title('K-Means Clustering (RP)')

    plt.subplot(2, 3, 5)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['EM_Cluster_Original'], palette='viridis')
    plt.title('EM Clustering (Original Space)')

    plt.subplot(2, 3, 6)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['EM_Cluster_PCA'], palette='viridis')
    plt.title('EM Clustering (PCA)')

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    sns.scatterplot(x=X_ica[:, 0], y=X_ica[:, 1], hue=df['EM_Cluster_ICA'], palette='viridis')
    plt.title('EM Clustering (ICA)')

    plt.subplot(2, 3, 2)
    sns.scatterplot(x=X_rp[:, 0], y=X_rp[:, 1], hue=df['EM_Cluster_RP'], palette='viridis')
    plt.title('EM Clustering (RP)')

    plt.show()

    # Calculate Adjusted Rand Index to determine the coefficient of similarity
    scores = {}
    clustering_methods = ['KMeans_Cluster_Original', 'KMeans_Cluster_PCA', 'KMeans_Cluster_ICA', 'KMeans_Cluster_RP',
                          'EM_Cluster_Original', 'EM_Cluster_PCA', 'EM_Cluster_ICA', 'EM_Cluster_RP']
    
    for method1 in clustering_methods:
        for method2 in clustering_methods:
            if method1 != method2:
                score = adjusted_rand_score(df[method1], df[method2])
                scores[(method1, method2)] = score
    
    # Print the scores
    for methods, score in scores.items():
        print(f"Adjusted Rand Index between {methods[0]} and {methods[1]}: {score}")

credit_card_clustering()
