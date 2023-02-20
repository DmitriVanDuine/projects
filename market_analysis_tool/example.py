# Import required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

# Load the dataset
df = pd.read_csv("AAPL.csv")

# Clean and preprocess the data
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Define the function to preprocess the data
def preprocess_data(df, target_column, n_steps):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X = []
    y = []

    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        y.append(scaled_data[i][target_column])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], df.shape[1]))

    return X, y, scaler

# Set the target column and number of time steps
target_column = 3
n_steps = 60

# Preprocess the data
X, y, scaler = preprocess_data(df.values, target_column, n_steps)

# Define the deep learning model architecture
def create_model(n_layers, n_nodes, dropout_rate, learning_rate):
    model = tf.keras.models.Sequential()

    for i in range(n_layers):
        if i == 0:
            model.add(tf.keras.layers.LSTM(n_nodes, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        elif i == n_layers-1:
            model.add(tf.keras.layers.LSTM(n_nodes))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        else:
            model.add(tf.keras.layers.LSTM(n_nodes, return_sequences=True))
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    return model

# Define the function to train the deep learning model
def train_model(n_layers, n_nodes, dropout_rate, learning_rate, epochs):
    model = create_model(int(n_layers), int(n_nodes), dropout_rate, learning_rate)
    model.fit(X, y, epochs=int(epochs), batch_size=32, verbose=0)
    return model

# Set the Bayesian optimization bounds
pbounds = {
    "n_layers": (1, 3),
    "n_nodes": (50, 200),
    "dropout_rate": (0.1, 0.5),
    "learning_rate": (1e-5, 1e-2),
    "epochs": (10, 100)
}

# Define the function to optimize the deep learning model using Bayesian optimization
def optimize_model(n_calls):
    optimizer = BayesianOptimization(f=train_model, pbounds=pbounds, verbose=2)
    optimizer.maximize(n_calls=n_calls)

    return optimizer

# Train the deep learning model using Bayesian optimization
n_calls = 10
optimizer = optimize_model(n_calls)

# Get the best hyperparameters and train the final model
best_params = optimizer.max["params"]
model = train_model(best_params["n_layers"], best_params["n_nodes"], best_params["dropout_rate"],
                    best_params["learning_rate"], best_params["epochs"])

def predict_future_prices(model, df, start_date, end_date, lookback_days, forecast_days):
    """
    Predict future prices using a trained model and input data.

    Args:
    model (sklearn.linear_model): The trained linear regression model.
    df (pandas.DataFrame): The input data.
    start_date (str): The start date of the prediction period.
    end_date (str): The end date of the prediction period.
    lookback_days (int): The number of days to look back when making a prediction.
    forecast_days (int): The number of days to forecast.

    Returns:
    A pandas DataFrame containing the predicted prices.
    """
    # Select the date range to predict
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize the output DataFrame
    predicted_prices = pd.DataFrame(index=date_range, columns=['predicted_price'])

    # Loop through each day in the date range
    for date in date_range:
        # Get the historical data for the lookback period
        start_idx = df.index.get_loc(date - pd.Timedelta(days=lookback_days), method='nearest')
        end_idx = df.index.get_loc(date - pd.Timedelta(days=1), method='nearest')
        X = df.iloc[start_idx:end_idx+1]

        # Predict the price for the current day
        y_pred = model.predict(X)

        # Add the predicted price to the output DataFrame
        predicted_prices.loc[date] = y_pred

        # Add the predicted price to the input DataFrame for use in the next iteration
        new_row = pd.DataFrame({'price': y_pred}, index=[date])
        df = pd.concat([df, new_row])

    # Add the forecasted prices to the output DataFrame
    last_date = predicted_prices.index[-1]
    forecast_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecasted_prices = pd.DataFrame(index=forecast_range, columns=['predicted_price'])
    forecasted_prices['predicted_price'] = model.predict(df.tail(lookback_days))
    predicted_prices = predicted_prices.append(forecasted_prices)

    return predicted_prices


# Bayesian optimization
from skopt import gp_minimize

def objective(params):
    """
    Objective function for Bayesian optimization.

    Args:
    params (list): The hyperparameters to optimize.

    Returns:
    The negative mean squared error of the model on the validation set.
    """
    # Convert the params list to a dictionary
    param_dict = {
        'n_estimators': int(params[0]),
        'max_depth': int(params[1]),
        'min_samples_split': int(params[2]),
        'min_samples_leaf': int(params[3])
    }

    # Train a random forest model with the given hyperparameters
    rf = RandomForestRegressor(random_state=42, **param_dict)
    rf.fit(X_train, y_train)

    # Calculate the mean squared error on the validation set
    _pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    # Return the negative mse (to maximize)
    return -mse

# Define the hyperparameter search space
space = [(10, 1000),  # n_estimators
         (1, 20),  # max_depth
         (2, 20),  # min_samples_split
         (1, 20)]  # min_samples_leaf

# Bayesian optimization
optimizer = BayesianOptimization(
    f=train_model,
    pbounds={"n_units": (10, 1000), "dropout_rate": (0.0, 0.5), "lr": (0.0001, 0.01)},
    verbose=2,
    random_state=1,
)
optimizer.maximize(n_iter=10)

# Print the optimal hyperparameters
best_params = optimizer.max["params"]
print(f"Optimal hyperparameters: {best_params}")

# Train the model using the optimal hyperparameters
model = build_model(best_params["n_units"], best_params["dropout_rate"], best_params["lr"])
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Reinforcement Learning
env = StockTradingEnvironment(df_train, df_val, df_test, window_size=10)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32
EPISODES = 10
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e + 1}/{EPISODES}, score: {info['total_profit']}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Evaluate the trained agent on the test set
state = env.reset()
state = np.reshape(state, [1, state_size])
for time in range(len(df_test)):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    state = next_state
    if done:
        break
print(f"Test set total profit: {info['total_profit']}")

# Deep Learning
# Build the model architecture
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=10))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test MAE: {test_mae}")

