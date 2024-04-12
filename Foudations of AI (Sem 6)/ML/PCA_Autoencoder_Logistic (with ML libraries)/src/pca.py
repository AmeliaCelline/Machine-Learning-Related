import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%  
        #centered it first
        self.mean = np.mean(X, axis = 0)
        centered = X - self.mean

        #do svd to get eigenfactors
        u, s, vh = np.linalg.svd(centered, full_matrices=True)
        # notes:
        # The right singular vectors in V are sorted in decreasing order of importance.
        # The rows of vh are the eigenvectors of Ah A and the columns of u are the eigenvectors of A Ah
    
        #take the first n_components eigenfactors
        self.components = vh[:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        ans = np.dot((X - self.mean), self.components.T)
        return ans

    def reconstruct(self, X):
        #TODO: 2%
        ans = np.dot(self.transform(X), self.components) + self.mean
        return ans

