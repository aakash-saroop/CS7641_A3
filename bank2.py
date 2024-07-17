import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_bank(df):
    df['Experience'] = df['Experience'].abs()
    return df

def find_optimal_pca_components(X):
    pca = PCA().fit(X)
    explained_variances = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variances >= 0.95) + 1
    return optimal_components, pca.explained_variance_ratio_

def find_optimal_ica_components(X):
    kurtosis_scores = []
    for i in range(2, min(X.shape[1], 20)):
        ica = FastICA(n_components=i, random_state=42)
        X_ica = ica.fit_transform(X)
        kurtosis_scores.append(np.mean(np.abs(kurtosis(X_ica, axis=0))))
    optimal_components = np.argmax(kurtosis_scores) + 2
    return optimal_components, kurtosis_scores

def apply_dimensionality_reduction_bank(X):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    optimal_pca_components, explained_variance_ratio = find_optimal_pca_components(X_scaled)
    pca = PCA(n_components=optimal_pca_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_pca = np.cumsum(pca.explained_variance_ratio_)
    
    # ICA
    optimal_ica_components, kurtosis_scores = find_optimal_ica_components(X_scaled)
    ica = FastICA(n_components=optimal_ica_components, random_state=42)
    X_ica = ica.fit_transform(X_scaled)
    kurtosis_ica = kurtosis(X_ica, axis=0)
    
    # Random Projection
    rp = GaussianRandomProjection(n_components=min(X.shape[1], 10), random_state=42)
    X_rp = rp.fit_transform(X_scaled)
    
    return X_scaled, X_pca, explained_variance_pca, explained_variance_ratio, X_ica, kurtosis_ica, kurtosis_scores, X_rp, optimal_pca_components, optimal_ica_components

def reconstruction_error(X, X_rp):
    rp = GaussianRandomProjection(n_components=X_rp.shape[1], random_state=42)
    X_projected = rp.fit_transform(X)
    X_reconstructed = np.linalg.pinv(rp.components_).dot(X_projected.T).T
    return mean_squared_error(X, X_reconstructed)

def evaluate_random_projections(X_scaled, n_runs=10):
    errors = []
    for _ in range(n_runs):
        errors.append(reconstruction_error(X_scaled, X_scaled))
    return errors

def add_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def analyze_and_visualize(X_scaled, X_pca, explained_variance_pca, explained_variance_ratio, X_ica, kurtosis_ica, kurtosis_scores, X_rp, optimal_pca_components, optimal_ica_components):
    # PCA Eigenvalue Distribution
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(1, len(explained_variance_ratio)+1), explained_variance_ratio, marker='o')
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Distribution by PCA')
    plt.grid(True)
    
    # ICA Kurtosis
    plt.subplot(1, 3, 2)
    plt.bar(np.arange(1, len(kurtosis_ica)+1), np.abs(kurtosis_ica), color='g')
    plt.xlabel('Independent Components')
    plt.ylabel('Kurtosis')
    plt.title('Absolute Kurtosis by ICA Components')
    plt.grid(True)
    
    # Random Projection
    plt.subplot(1, 3, 3)
    plt.scatter(X_rp[:, 0], X_rp[:, 1], marker='o', color='r', alpha=0.5)
    plt.xlabel('Random Projection Component 1')
    plt.ylabel('Random Projection Component 2')
    plt.title('Random Projection')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    # Print optimal components
    print(f'Optimal number of PCA components: {optimal_pca_components}')
    print(f'Optimal number of ICA components: {optimal_ica_components}')
    
    # Random Projection Reconstruction Error
    errors = evaluate_random_projections(X_scaled)
    plt.figure()
    plt.hist(errors, bins=10, color='blue', edgecolor='black')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error of Random Projections')
    plt.show()
    
    # Effect of Noise
    X_noisy = add_noise(X_scaled)
    pca_noisy = PCA(n_components=X_pca.shape[1], random_state=42).fit_transform(X_noisy)
    ica_noisy = FastICA(n_components=X_ica.shape[1], random_state=42).fit_transform(X_noisy)
    rp_noisy = GaussianRandomProjection(n_components=X_rp.shape[1], random_state=42).fit_transform(X_noisy)
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(pca_noisy[:, 0], pca_noisy[:, 1], alpha=0.5)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA with Noise')
    
    plt.subplot(1, 3, 2)
    plt.scatter(ica_noisy[:, 0], ica_noisy[:, 1], alpha=0.5, color='g')
    plt.xlabel('ICA Component 1')
    plt.ylabel('ICA Component 2')
    plt.title('ICA with Noise')
    
    plt.subplot(1, 3, 3)
    plt.scatter(rp_noisy[:, 0], rp_noisy[:, 1], alpha=0.5, color='r')
    plt.xlabel('RP Component 1')
    plt.ylabel('RP Component 2')
    plt.title('RP with Noise')
    
    plt.tight_layout()
    plt.show()
    
    # Rank of Data
    rank = np.linalg.matrix_rank(X_scaled)
    print(f'Rank of the data matrix: {rank}')
    
    # Collinearity of Data
    correlation_matrix = np.corrcoef(X_scaled, rowvar=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Collinearity Quantitatively
    print('Correlation matrix:\n', correlation_matrix)

# Example usage
df_bank = pd.read_csv('../data/bank.csv')
df_bank = preprocess_bank(df_bank)
X_bank = df_bank.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])

X_scaled_bank, X_pca_bank, explained_variance_pca_bank, explained_variance_ratio_bank, X_ica_bank, kurtosis_ica_bank, kurtosis_scores_bank, X_rp_bank, optimal_pca_components_bank, optimal_ica_components_bank = apply_dimensionality_reduction_bank(X_bank)
analyze_and_visualize(X_scaled_bank, X_pca_bank, explained_variance_pca_bank, explained_variance_ratio_bank, X_ica_bank, kurtosis_ica_bank, kurtosis_scores_bank, X_rp_bank, optimal_pca_components_bank, optimal_ica_components_bank)
