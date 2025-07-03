import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, CNN1D, Conv1D, MaxPooling1D, Flatten, Dense, 
    Dropout, BatchNormalization, Concatenate, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DowntimePredictionModel:
    def __init__(self, sequence_length=50, n_features=10):
        """
        Initialize the downtime prediction model
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of input features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        
    def create_model(self):
        """Create the hybrid CNN-LSTM model architecture"""
        
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # CNN Branch for spatial feature extraction
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        
        cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = Dropout(0.2)(cnn_branch)
        
        # LSTM Branch for temporal sequence modeling
        lstm_branch = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(input_layer)
        lstm_branch = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm_branch)
        lstm_branch = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm_branch)
        
        # Flatten CNN output
        cnn_flattened = Flatten()(cnn_branch)
        
        # Concatenate CNN and LSTM features
        combined = Concatenate()([cnn_flattened, lstm_branch])
        
        # Dense layers for final prediction
        dense = Dense(256, activation='relu')(combined)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        dense = Dense(128, activation='relu')(dense)
        dense = Dropout(0.2)(dense)
        
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(0.1)(dense)
        
        # Output layer - predicting downtime in seconds
        output = Dense(1, activation='linear', name='downtime_seconds')(dense)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def generate_sample_data(self, n_samples=10000):
        """
        Generate sample data for demonstration
        Features could include: CPU usage, memory usage, network I/O, disk I/O, 
        temperature, error counts, etc.
        """
        np.random.seed(42)
        
        # Generate features that correlate with downtime
        data = []
        targets = []
        
        for i in range(n_samples):
            # Generate sequence of system metrics
            sequence = np.random.randn(self.sequence_length, self.n_features)
            
            # Add some realistic patterns
            # CPU usage (0-100%)
            sequence[:, 0] = np.abs(sequence[:, 0] * 20 + 50)  # CPU usage
            sequence[:, 1] = np.abs(sequence[:, 1] * 15 + 30)  # Memory usage
            sequence[:, 2] = np.abs(sequence[:, 2] * 10 + 20)  # Disk I/O
            sequence[:, 3] = np.abs(sequence[:, 3] * 5 + 10)   # Network I/O
            sequence[:, 4] = np.abs(sequence[:, 4] * 3 + 45)   # Temperature
            
            # Error counts and other metrics
            sequence[:, 5] = np.abs(sequence[:, 5] * 2)        # Error count
            sequence[:, 6] = np.abs(sequence[:, 6] * 1.5)      # Warning count
            sequence[:, 7] = np.abs(sequence[:, 7] * 0.8 + 1)  # Response time
            sequence[:, 8] = np.abs(sequence[:, 8] * 0.5)      # Queue length
            sequence[:, 9] = np.abs(sequence[:, 9] * 0.3)      # Connection count
            
            # Calculate downtime based on system state
            # Higher values in critical metrics lead to longer downtime
            cpu_stress = np.mean(sequence[:, 0]) / 100
            memory_stress = np.mean(sequence[:, 1]) / 100
            error_impact = np.sum(sequence[:, 5]) * 10
            temp_impact = max(0, np.mean(sequence[:, 4]) - 60) * 2
            
            # Base downtime calculation (in seconds)
            base_downtime = (cpu_stress * 300 + memory_stress * 200 + 
                           error_impact + temp_impact + np.random.normal(0, 50))
            
            # Add some randomness and ensure positive values
            downtime = max(0, base_downtime + np.random.normal(0, 100))
            
            data.append(sequence)
            targets.append(downtime)
        
        return np.array(data), np.array(targets)
    
    def prepare_data(self, X, y):
        """Prepare and scale the data"""
        # Reshape X for scaling
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Scale targets
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the model"""
        # Prepare data
        X_scaled, y_scaled = self.prepare_data(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale input
        X_reshaped = X.reshape(-1, self.n_features)
        X_scaled = self.scaler_X.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten()
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, X, y, n_samples=100):
        """Plot predictions vs actual values"""
        y_pred = self.predict(X[:n_samples])
        
        plt.figure(figsize=(12, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y[:n_samples], y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Downtime (seconds)')
        plt.ylabel('Predicted Downtime (seconds)')
        plt.title('Predicted vs Actual Downtime')
        plt.grid(True)
        
        # Time series plot
        plt.subplot(1, 2, 2)
        indices = range(n_samples)
        plt.plot(indices, y[:n_samples], label='Actual', alpha=0.7)
        plt.plot(indices, y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Downtime (seconds)')
        plt.title('Time Series Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DowntimePredictionModel(sequence_length=50, n_features=10)
    
    # Generate sample data
    print("Generating sample data...")
    X, y = model.generate_sample_data(n_samples=5000)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Downtime range: {y.min():.2f} - {y.max():.2f} seconds")
    
    # Create and display model architecture
    print("\nCreating model...")
    model.create_model()
    model.model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.train(X, y, epochs=50, batch_size=32)
    
    # Plot training history
    model.plot_training_history()
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X[-1000:], y[-1000:])  # Use last 1000 samples for evaluation
    
    print(f"Evaluation Metrics:")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    
    # Plot predictions
    model.plot_predictions(X[-1000:], y[-1000:], n_samples=100)
    
    # Example prediction
    print("\nExample predictions:")
    sample_indices = np.random.choice(len(X), 5, replace=False)
    for i in sample_indices:
        pred = model.predict(X[i:i+1])
        actual = y[i]
        print(f"Sample {i}: Predicted={pred[0]:.2f}s, Actual={actual:.2f}s, Error={abs(pred[0]-actual):.2f}s")
