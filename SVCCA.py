import torch
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from torch.linalg import svd as torch_svd

class SVCCA:
    def __init__(self, hidden_states, labels, use_gpu=False):
        self.hidden_states = hidden_states
        self.Ys = labels
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.max_correlations = []

    def compute_svd(self, X, threshold=0.99):
        X = X.astype(np.float32)
        if self.use_gpu:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            U, S, Vt = torch_svd(X_tensor, full_matrices=False)
            U, S, Vt = U.cpu().numpy(), S.cpu().numpy(), Vt.cpu().numpy()
        else:
            try:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
            except np.linalg.LinAlgError:
                # Fallback to a more stable SVD implementation
                U, S, Vt = np.linalg.svd(X + np.random.normal(0, 1e-10, X.shape), full_matrices=False)
        
        cumulative_variance = np.cumsum(S) / np.sum(S)
        num_components = np.searchsorted(cumulative_variance, threshold) + 1
        return U[:, :num_components] @ np.diag(S[:num_components]), num_components, Vt[:num_components]

    def compute_cca_cpu(self, X, Y, num_components):
        cca = CCA(n_components=num_components)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        return X_c, Y_c, cca

    def compute_cca_gpu(self, X, Y, num_components):
        X_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        Y_tensor = torch.tensor(Y, dtype=torch.float64, device=self.device)

        X = X_tensor - X_tensor.mean(dim=0, keepdim=True)
        Y = Y_tensor - Y_tensor.mean(dim=0, keepdim=True)
        
        # Compute covariance matrices
        C_XX = torch.matmul(X.T, X) / (X.size(0) - 1)
        C_YY = torch.matmul(Y.T, Y) / (Y.size(0) - 1)
        C_XY = torch.matmul(X.T, Y) / (X.size(0) - 1)

        # Compute the inverses of covariance matrices
        inv_C_XX = torch.inverse(C_XX + torch.eye(C_XX.size(0), device=self.device) * 1e-5)
        inv_C_YY = torch.inverse(C_YY + torch.eye(C_YY.size(0), device=self.device) * 1e-5)

        # Compute the matrix for the eigenvalue problem
        M = torch.matmul(torch.matmul(inv_C_XX, C_XY), torch.matmul(inv_C_YY, C_XY.T))

        # Solve the eigenvalue problem using torch.linalg.eigh
        eigvals, eigvecs = torch.linalg.eigh(M, UPLO='L')
        
        # Select the top num_components eigenvectors
        A = eigvecs[:, -num_components:]
        
        # Compute the canonical variables
        U = torch.matmul(X, A)
        V = torch.matmul(Y, torch.matmul(torch.matmul(inv_C_YY, C_XY.T), A))
        
        return U.cpu().numpy(), V.cpu().numpy(), eigvals.cpu().numpy()

    def svcca(self, X, Y, threshold=0.99):
        X_svd, num_components_X, Vt_X = self.compute_svd(X, threshold)
        Y_svd, num_components_Y, Vt_Y = self.compute_svd(Y, threshold)
        
        num_components = min(num_components_X, num_components_Y)
        if self.use_gpu:
            X_cca, Y_cca, cca_model = self.compute_cca_gpu(X_svd, Y_svd, num_components)
        else:
            X_cca, Y_cca, cca_model = self.compute_cca_cpu(X_svd, Y_svd, num_components)
        
        return X_cca, Y_cca, cca_model, Vt_X, Vt_Y

    def perform_svcca(self, hidden_states, labels):
        # Normalize hidden states and labels
        scaler = StandardScaler()
        hidden_states_normalized = scaler.fit_transform(hidden_states).astype(np.float32)
        labels_normalized = scaler.fit_transform(labels).astype(np.float32)
        
        # Perform SVCCA
        X_cca, Y_cca, cca_model, Vt_X, Vt_Y = self.svcca(hidden_states_normalized, labels_normalized)
        corr = np.corrcoef(X_cca.T, Y_cca.T)
        return corr, cca_model, Vt_X, Vt_Y

    def calculate_max_correlations(self):
        for i in range(len(self.hidden_states)):
            correlation, cca_model, Vt_X, Vt_Y = self.perform_svcca(self.hidden_states[i], self.Ys[i])
            max_corr = np.max(correlation[:2, 2:])
            self.max_correlations.append(max_corr)
            print(f"Layer {i}: Max Corr = {max_corr}")

    def save_results(self, file_path: str):
        results_df = pd.DataFrame({
            'layer': range(len(self.hidden_states)),
            'Correlation': self.max_correlations,
        })
        # Save results to CSV file
        results_df.to_csv(file_path, index=False)
        print(f"Ergebnisse erfolgreich in '{file_path}' gespeichert.")

if __name__ == "__main__":
    # Simulate Hidden States and Labels
    hidden_states = [np.random.randn(100, 1024) for _ in range(12)]  # Hidden States for 12 layers
    Ys = [np.eye(10)[np.random.randint(0, 10, 100)] for _ in range(12)]  # One-Hot-encoded labels for 12 layers
    svcca = SVCCA(hidden_states, Ys, use_gpu=True) # Initialize SVCCA object
    svcca.calculate_max_correlations() # Calculate max correlations
