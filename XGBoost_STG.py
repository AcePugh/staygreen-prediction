import pandas as pd  # Import the pandas library for data manipulation
from sklearn.model_selection import train_test_split  # Import the function to split the data into training and testing sets
from sklearn.metrics import r2_score, mean_squared_error  # Import metrics to evaluate the model
from math import sqrt  # Import sqrt to calculate the root mean squared error
import xgboost as xgb  # Import the XGBoost library
import tkinter as tk  # Import tkinter for GUI elements
from tkinter import filedialog, simpledialog, messagebox  # Import specific GUI components for file and input dialogs

def get_hyperparameters():
    try:
        learning_rate = simpledialog.askfloat("Input", "Enter Learning Rate (e.g., 0.1):", minvalue=0.01, maxvalue=1.0)  # Ask the user for the learning rate
        if learning_rate is None:  # Check if the dialog box was cancelled
            return None
        max_depth = simpledialog.askinteger("Input", "Enter Max Depth (e.g., 6):", minvalue=1)  # Ask the user for the maximum depth
        if max_depth is None:  # Check if the dialog box was cancelled
            return None
        min_child_weight = simpledialog.askinteger("Input", "Enter Minimum Child Weight (e.g., 1):", minvalue=1)  # Ask the user for the minimum child weight
        if min_child_weight is None:  # Check if the dialog box was cancelled
            return None
    except Exception as e:  # Catch any exceptions that occur during input
        messagebox.showerror("Input Error", f"An error occurred: {e}")  # Display an error message
        return None
    return learning_rate, max_depth, min_child_weight  # Return the hyperparameters

def load_data():
    root = tk.Tk()  # Create a new tkinter window
    root.withdraw()  # Hide the tkinter window
    file_path = filedialog.askopenfilename()  # Open a file dialog to select a CSV file
    if not file_path:  # Check if the file dialog was cancelled
        messagebox.showinfo("File Selection", "File selection cancelled.")  # Show a cancellation message
        return None
    try:
        df = pd.read_csv(file_path)  # Attempt to read the CSV file into a DataFrame
    except Exception as e:  # Catch file reading errors
        messagebox.showerror("File Load Error", f"Error loading file: {e}")  # Display an error message
        return None
    return df  # Return the loaded DataFrame

def calculate_adjusted_r2(X, y, y_pred):
    n = len(y)  # Number of samples
    p = X.shape[1]  # Number of features
    r2 = r2_score(y, y_pred)  # Calculate the R² score
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # Calculate the adjusted R² score
    return adj_r2  # Return the adjusted R² score

def save_predictions(df, y_train, y_train_pred, y_test, y_test_pred):
    results = pd.DataFrame({  # Create a DataFrame to store the results
        'Plot Number': df.loc[y_train.index.union(y_test.index), 'plot'],  # Add the plot numbers
        'Actual Staygreen': pd.concat([y_train, y_test]),  # Adding the actual staygreen values
        'Predicted Staygreen': pd.concat([pd.Series(y_train_pred, index=y_train.index), pd.Series(y_test_pred, index=y_test.index)]),  # Add the predicted staygreen values
        'Set': pd.concat([pd.Series(['Train'] * len(y_train), index=y_train.index), pd.Series(['Test'] * len(y_test), index=y_test.index)])  # Add the set (training or testing)
    })
    save_path = filedialog.asksaveasfilename(title="Save the results as", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])  # Open a file dialog to save the results
    if save_path:  # Check if a path is provided
        results.to_csv(save_path, index=False)  # Save the DataFrame to a CSV file
        messagebox.showinfo("Save Successful", f"Results saved to {save_path}")  # Show a confirmation message
    else:  # If the save dialog was cancelled
        messagebox.showinfo("Save Cancelled", "Results save cancelled.")  # Show a cancellation message

def main():
    df = load_data()  # Load the data
    if df is None:  # Check if the data was loaded successfully
        return

    params = get_hyperparameters()  # Get the hyperparameters from the user
    if params is None:  # Check if the hyperparameters were provided
        messagebox.showinfo("Parameter Input", "Parameter input cancelled or invalid.")  # Show a cancellation message
        return

    learning_rate, max_depth, min_child_weight = params  # Unpack the hyperparameters

    cols = [col for col in df.columns if col not in ['plot', 'staygreen']]  # Exclude 'plot' and 'staygreen' columns from features
    X = df[cols]  # Define the features
    y = df['staygreen']  # Define the target variable

    random_state = 186  # Set a random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)  # Split the data into training and testing sets
    model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight, random_state=random_state)  # Create the XGBoost regressor model
    model.fit(X_train, y_train)  # Fit the model on the training data

    y_train_pred = model.predict(X_train)  # Predict on the training set
    y_test_pred = model.predict(X_test)  # Predict on the testing set

    adj_train_r2 = calculate_adjusted_r2(X_train, y_train, y_train_pred)  # Calculate the adjusted R² for the training set
    adj_test_r2 = calculate_adjusted_r2(X_test, y_test, y_test_pred)  # Calculate the adjusted R² for the testing set
    mse_train = mean_squared_error(y_train, y_train_pred)  # Calculate the mean squared error for the training set
    mse_test = mean_squared_error(y_test, y_test_pred)  # Calculate the mean squared error for the testing set
    rmse_train = sqrt(mse_train)  # Calculate the root mean squared error for the training set
    rmse_test = sqrt(mse_test)  # Calculate the root mean squared error for the testing set

    print(f"Training Adjusted R²: {adj_train_r2}")  # Print the adjusted R² for the training set
    print(f"Test Adjusted R²: {adj_test_r2}")  # Print the adjusted R² for the testing set
    print(f"Training MSE: {mse_train}")  # Print the mean squared error for the training set
    print(f"Test MSE: {mse_test}")  # Print the mean squared error for the testing set
    print(f"Training RMSE: {rmse_train}")  # Print the root mean squared error for the training set
    print(f"Test RMSE: {rmse_test}")  # Print the root mean squared error for the testing set

    save_predictions(df, y_train, y_train_pred, y_test, y_test_pred)  # Save the predictions

if __name__ == "__main__":
    main()  # Run the main function
