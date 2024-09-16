import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def get_hyperparameters():
    try:
        n_estimators = simpledialog.askinteger("Input", "Enter Number of Estimators (e.g., 100):", minvalue=1)
        if n_estimators is None:
            return None
        max_depth = simpledialog.askinteger("Input", "Enter Max Depth (e.g., 6):", minvalue=1)
        if max_depth is None:
            return None
    except Exception as e:
        messagebox.showerror("Input Error", f"An error occurred: {e}")
        return None
    return n_estimators, max_depth

def load_data():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if not file_path:
        messagebox.showinfo("File Selection", "File selection cancelled.")
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("File Load Error", f"Error loading file: {e}")
        return None
    return df

def calculate_adjusted_r2(X, y, y_pred):
    n = len(y)
    p = X.shape[1]
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2

def save_predictions(df, y_train, y_train_pred, y_test, y_test_pred):
    results = pd.DataFrame({
        'Plot Number': df.loc[y_train.index.union(y_test.index), 'plot'],
        'Actual Staygreen': pd.concat([y_train, y_test]),
        'Predicted Staygreen': pd.concat([pd.Series(y_train_pred, index=y_train.index), pd.Series(y_test_pred, index=y_test.index)]),
        'Set': pd.concat([pd.Series(['Train'] * len(y_train), index=y_train.index), pd.Series(['Test'] * len(y_test), index=y_test.index)])
    })
    save_path = filedialog.asksaveasfilename(title="Save the results as", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_path:
        results.to_csv(save_path, index=False)
        messagebox.showinfo("Save Successful", f"Results saved to {save_path}")
    else:
        messagebox.showinfo("Save Cancelled", "Results save cancelled.")

def main():
    df = load_data()
    if df is None:
        return

    params = get_hyperparameters()
    if params is None:
        messagebox.showinfo("Parameter Input", "Parameter input cancelled or invalid.")
        return

    n_estimators, max_depth = params

    cols = [col for col in df.columns if col not in ['plot', 'staygreen']]
    X = df[cols]
    y = df['staygreen']

    random_state = 186
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    adj_train_r2 = calculate_adjusted_r2(X_train, y_train, y_train_pred)
    adj_test_r2 = calculate_adjusted_r2(X_test, y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = sqrt(mse_train)
    rmse_test = sqrt(mse_test)

    print(f"Training Adjusted R²: {adj_train_r2}")
    print(f"Test Adjusted R²: {adj_test_r2}")
    print(f"Training MSE: {mse_train}")
    print(f"Test MSE: {mse_test}")
    print(f"Training RMSE: {rmse_train}")
    print(f"Test RMSE: {rmse_test}")

    save_predictions(df, y_train, y_train_pred, y_test, y_test_pred)

if __name__ == "__main__":
    main()
