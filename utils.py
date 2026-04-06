import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split

def generate_dataset(dataset_name, n_samples=400, noise=0.15, random_state=42):
    """
    Generates educational datasets for classification.
    Returns transposed data shapes suitable for standard NN math implementation:
    X: (n_features, m_samples), y: (1, m_samples)
    """
    if "Linear" in dataset_name:
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=random_state, 
                                   n_clusters_per_class=1)
        if noise > 0:
            np.random.seed(random_state)
            X += np.random.randn(*X.shape) * noise
            
    elif "Moons" in dataset_name:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        
    else: # Circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        
    # Standard splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Transposing to align with mathematical conventions where columns are samples
    return X_train.T, X_test.T, y_train.reshape(1, -1), y_test.reshape(1, -1)
