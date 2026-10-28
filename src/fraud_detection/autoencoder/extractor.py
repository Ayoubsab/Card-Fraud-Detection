import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from fraud_detection.autoencoder.network import AutoEncoder

logger = logging.getLogger(__name__)


class AutoencoderFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, epochs=20, batch_size=256, learning_rate=0.001, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        We only train the Autoencoder on NON-FRAUD data
        so it learns exactly what a normal transaction looks like.
        """
        torch.manual_seed(self.random_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_fit = X
        if y is not None:
            y_series = pd.Series(y, index=X.index) if isinstance(y, np.ndarray) else y
            X_fit = X[y_series == 0]

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_fit)

        X_tensor = torch.FloatTensor(X_scaled).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model_ = AutoEncoder(input_dim=X.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        logger.info("Training autoencoder on %d legitimate transactions (%s, %d epochs)",
                    len(X_fit), device, self.epochs)
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Log progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                logger.info("Epoch %d/%d | Loss: %.4f", epoch + 1, self.epochs, total_loss / len(dataloader))

        # Keep the fitted model on CPU so the saved pipeline loads on machines without a GPU
        self.model_ = self.model_.cpu().eval()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """
        Pass all data through the trained model (in batches, to bound memory)
        to calculate the Reconstruction Error.
        """
        check_is_fitted(self, 'model_')
        X_copy = X.copy()

        X_tensor = torch.FloatTensor(self.scaler_.transform(X_copy))

        errors = []
        with torch.no_grad():
            for batch_x in torch.split(X_tensor, 8192):
                reconstructed = self.model_(batch_x)
                # Mean Squared Error (MSE) per row
                errors.append(torch.mean((batch_x - reconstructed) ** 2, dim=1))

        # Add the anomaly score as a brand new feature for XGBoost
        X_copy['ae_reconstruction_error'] = torch.cat(errors).numpy()

        return X_copy
