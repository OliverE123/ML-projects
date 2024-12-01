import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta

# Define LSTM model
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Dataset class
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.seq_length],
            self.data[idx + self.seq_length][0]  
        )

# Prepare data
def prepare_data(ticker, seq_length):
    # Fetch data from Yahoo Finance
    df = yf.download(ticker, start="2014-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df = df.dropna()

    # Scale data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[['Close', 'MA20', 'MA50', 'RSI', 'Volatility']])

    # Split into train and test sets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data, scaler, df

# Compute RSI
def compute_rsi(series, window):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Save and load model weights
def save_model(model, ticker):
    torch.save(model.state_dict(), f"{ticker}.pth")

def load_model(ticker, input_size, hidden_size, num_layers):
    model = StockPredictor(input_size, hidden_size, num_layers, 1)
    if os.path.exists(f"{ticker}.pth"):
        model.load_state_dict(torch.load(f"{ticker}.pth"))
        print(f"Loaded saved weights for {ticker}.")
    return model

# Train model
def train_model(train_data, seq_length, input_size, hidden_size, num_layers, lr, epochs, ticker):
    train_dataset = StockDataset(train_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = load_model(ticker, input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for seq, target in train_loader:
            seq, target = seq.float(), target.float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    save_model(model, ticker)
    return model

# Predict function
def predict(model, data, seq_length, scaler):
    model.eval()
    inputs = torch.tensor(data[-seq_length:]).float().unsqueeze(0)  
    predictions = []

    for _ in range(10):  # Predict 10 steps into the future
        output = model(inputs)  
        predictions.append(output.item())  

        # Expand `output` to match the feature dimension of `inputs`
        output = output.detach().view(1, 1, 1)  
        output = output.repeat(1, 1, inputs.shape[2])  

        # Update `inputs` by appending `output` and removing the oldest timestep
        inputs = torch.cat((inputs[:, 1:, :], output), dim=1)  

    # Transform predictions back to the original scale
    feature_count = scaler.min_.shape[0]  
    predictions_padded = [[p] + [0] * (feature_count - 1) for p in predictions]
    predictions = scaler.inverse_transform(predictions_padded)[:, 0]  

    return predictions

# Manage prediction files
def save_predictions(predictions, ticker):
    today = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{ticker}_predictions_{today}.csv"
    pd.DataFrame(predictions, columns=["Predicted Prices"]).to_csv(file_name, index=False)
    print(f"Predictions saved to {file_name}")

def load_predictions(ticker):
    today = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{ticker}_predictions_{today}.csv"
    if os.path.exists(file_name):
        print(f"Loading predictions from {file_name}")
        return pd.read_csv(file_name)["Predicted Prices"].tolist()
    return None

def delete_old_predictions(ticker):
    cutoff_date = datetime.now() - timedelta(days=2)
    for file in os.listdir():
        if file.startswith(ticker) and file.endswith(".csv"):
            try:
                date_str = file.split("_predictions_")[1].split(".csv")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff_date:
                    os.remove(file)
                    print(f"Deleted old prediction file: {file}")
            except (IndexError, ValueError):
                pass

# Plot results
# Plot results with full historical data and predictions
def plot_results(df, predictions):
    plt.figure(figsize=(12, 6))
    
    # Plot all historical data
    plt.plot(df.index, df['Close'], label='Historical Data', color='blue')
    
    # Generate future dates for predictions
    future_dates = pd.date_range(df.index[-1], periods=len(predictions) + 1, freq='B')[1:]
    
    # Plot predictions
    plt.plot(future_dates, predictions, label='Predicted Data', linestyle='-', color='red')
    
    # Add labels and legend
    plt.title("Stock Price Prediction with Historical Data")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

# Main function
def main():
    ticker = input("Enter the stock ticker symbol: ")
    seq_length = 60
    input_size = 5  # Close, MA20, MA50, RSI, Volatility
    hidden_size = 50
    num_layers = 2
    lr = 0.001
    epochs = 30

    delete_old_predictions(ticker)  # Clean up old predictions

    predictions = load_predictions(ticker)  # Check for existing predictions
    if predictions is None:
        print("No existing predictions found. Computing new predictions...A")
        train_data, test_data, scaler, df = prepare_data(ticker, seq_length)
        model = train_model(train_data, seq_length, input_size, hidden_size, num_layers, lr, epochs, ticker)
        predictions = predict(model, test_data, seq_length, scaler)
        save_predictions(predictions, ticker)
    else:
        _, _, _, df = prepare_data(ticker, seq_length)  # Only reload data for plotting

    plot_results(df, predictions)

if __name__ == "__main__":
    main()