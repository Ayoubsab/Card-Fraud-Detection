import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from .AutoEncoder import AutoEncoder

class AutoencoderFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, epochs=20, batch_size=256, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y=None):
        """
        We only train the Autoencoder on NON-FRAUD data
        so it learns exactly what a normal transaction looks like.
        """

        X_fit = X.copy()

        if y is not None:
            y_series = pd.Series(y, index=X_fit.index) if isinstance(y, np.ndarray) else y
            X_fit = X_fit[y_series == 0]
            print(f"Training AE on {len(X_fit)} legitimate transactions...")
            
        X_scaled = self.scaler.fit_transform(X_fit)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)


        self.model = AutoEncoder(input_dim=X.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"Training Autoencoder on {self.device} for {self.epochs} epochs...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {total_loss/len(dataloader):.4f}")

        return self
    
    def transform(self, X):
        """
        Pass all data through the trained model to calculate 
        the Reconstruction Error.
        """
        X_copy = X.copy()
        
        X_scaled = self.scaler.transform(X_copy)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            
            # Calculate Mean Squared Error (MSE) per row
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            
        # Add the anomaly score as a brand new feature for XGBoost
        X_copy['ae_reconstruction_error'] = mse
        
        return X_copy