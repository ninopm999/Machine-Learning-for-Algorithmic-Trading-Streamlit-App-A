import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Helper Functions ---

def simulate_stock_data(num_days=365, initial_price=100):
    """
    Simulates historical stock price data with some randomness and trend.
    """
    np.random.seed(42) # for reproducibility
    # Corrected: Changed pd.date_date_range to pd.date_range
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    prices = [initial_price]
    for _ in range(1, num_days):
        # Simulate daily change with some trend and noise
        change = np.random.normal(0, 1.5) + (prices[-1] - initial_price) * 0.001
        new_price = prices[-1] + change
        # Ensure price doesn't go too low
        prices.append(max(10, new_price))
    
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    df.set_index('Date', inplace=True)
    return df

def feature_engineer(df):
    """
    Creates simple technical indicator-like features from the 'Close' price.
    """
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI_dummy'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).rolling(window=14).mean() # A very simplified RSI-like
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # Target variable: next day's price movement (1 if up, 0 if down/same)
    df['Target_Movement'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True) # Remove rows with NaN values introduced by rolling windows and shift
    return df

def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    """
    Generates predictions using the trained model.
    """
    predictions = model.predict(X_test)
    return predictions

def apply_trading_strategy(df_with_predictions):
    """
    Applies a simple trading strategy:
    - Buy if predicted movement is 'up' (e.g., prediction > 0.5 for probability of up)
    - Sell/Hold if predicted movement is 'down' or 'same'
    """
    # For RandomForestRegressor predicting a target movement probability (0 or 1)
    # we can interpret the prediction as a probability-like score.
    # If the model predicts a value closer to 1, it means 'up'.
    df_with_predictions['Signal'] = np.where(df_with_predictions['Predicted_Movement'] > 0.5, 'BUY', 'HOLD/SELL')
    return df_with_predictions

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Algorithmic Trading ML Demo")

st.title("📈 Machine Learning for Algorithmic Trading and Market Prediction")
st.markdown("""
This application demonstrates a highly simplified machine learning approach for predicting stock price movements and generating trading signals.
**Disclaimer:** This is for educational purposes only and uses simulated data. It is not financial advice, and actual trading involves significant risk.
""")

# --- Data Generation Section ---
st.header("1. Simulate Stock Data")
num_days_input = st.slider("Select number of historical days to simulate:", 100, 1000, 365)
initial_price_input = st.slider("Select initial stock price:", 50, 200, 100)

if st.button("Generate Data"):
    st.session_state.df_raw = simulate_stock_data(num_days=num_days_input, initial_price=initial_price_input)
    st.success(f"Simulated {num_days_input} days of stock data.")

if 'df_raw' in st.session_state:
    st.subheader("Simulated Stock Prices")
    st.line_chart(st.session_state.df_raw['Close'])
    st.write(st.session_state.df_raw.tail())

    # --- Feature Engineering & Model Training Section ---
    st.header("2. Feature Engineering & Model Training")
    st.markdown("Features like Simple Moving Averages (SMA) and a dummy RSI are engineered.")
    
    if st.button("Engineer Features & Train Model"):
        df_features = feature_engineer(st.session_state.df_raw.copy())
        
        # Define features (X) and target (y)
        features = ['SMA_5', 'SMA_20', 'RSI_dummy', 'Volatility']
        target = 'Target_Movement'
        
        X = df_features[features]
        y = df_features[target]
        
        # Split data into training and testing sets
        # Use a time-series split to avoid data leakage (train on earlier data, test on later)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        st.session_state.model = train_model(X_train, y_train)
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.df_features = df_features # Store for later use
        
        st.success("Features engineered and model trained successfully!")
        
        # Evaluate training performance (optional, for demonstration)
        train_preds = st.session_state.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        st.write(f"Model Training RMSE (Target Movement): {train_rmse:.4f}")

    # --- Prediction & Strategy Section ---
    if 'model' in st.session_state and 'X_test' in st.session_state:
        st.header("3. Generate Predictions & Trading Signals")
        
        if st.button("Generate Predictions & Signals"):
            predictions = make_predictions(st.session_state.model, st.session_state.X_test)
            
            # Create a DataFrame for results
            results_df = pd.DataFrame({
                'Actual_Close': st.session_state.df_raw['Close'].loc[st.session_state.X_test.index],
                'Predicted_Movement': predictions,
                'Actual_Movement': st.session_state.y_test
            })
            
            st.session_state.trading_signals_df = apply_trading_strategy(results_df.copy())
            st.success("Predictions generated and trading signals applied!")

            st.subheader("Predicted Movements and Trading Signals (Test Data)")
            st.write(st.session_state.trading_signals_df.head(10))

            # --- Visualization of Signals ---
            st.subheader("Visualization of Trading Signals")
            fig = plt.figure(figsize=(12, 6))
            plt.plot(st.session_state.trading_signals_df.index, st.session_state.trading_signals_df['Actual_Close'], label='Actual Close Price', color='blue')
            
            buy_signals = st.session_state.trading_signals_df[st.session_state.trading_signals_df['Signal'] == 'BUY']
            plt.scatter(buy_signals.index, buy_signals['Actual_Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
            
            plt.title('Stock Price with Trading Signals')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            st.pyplot(fig)

            st.markdown("""
            **Interpretation of Trading Signals:**
            * **Green Triangles (^):** Indicate a 'BUY' signal generated by the model based on the predicted upward movement.
            * **No Mark:** Indicates a 'HOLD/SELL' signal.
            """)
            st.write("---")
            st.info("""
            **Next Steps & Considerations for a Real System:**
            * **Real Data:** Integrate with financial APIs (e.g., Yahoo Finance, Alpha Vantage) for live/historical data.
            * **Advanced Features:** Incorporate more sophisticated technical indicators (MACD, Bollinger Bands), volume data, sentiment analysis, macroeconomic indicators.
            * **Sophisticated Models:** Experiment with time series models (LSTM, ARIMA), deep learning architectures, or even reinforcement learning for trading.
            * **Backtesting:** Rigorously test the strategy on historical data to evaluate performance metrics like Sharpe Ratio, drawdown, and total returns.
            * **Risk Management:** Implement stop-loss orders, position sizing, and portfolio diversification.
            * **Execution:** Connect to a brokerage API for automated trade execution.
            * **Deployment:** Deploy on a robust cloud platform for continuous operation.
            """)
