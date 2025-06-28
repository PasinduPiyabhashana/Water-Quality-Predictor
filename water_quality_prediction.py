import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os

class WaterQualityPredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Water Quality Predictor")
        master.geometry("600x700")
        master.configure(bg='#f5f5f5')

        # Load the pre-trained model and scalers
        self.load_model_and_scalers()

        # Create UI Components
        self.create_ui()

    def load_model_and_scalers(self):
        # Determine the base path (project directory)
        base_path = os.path.dirname(os.path.abspath(__file__))

        try:
            # Load model
            model_path = os.path.join(base_path, 'water_quality_model.h5')
            self.model = tf.keras.models.load_model(model_path)

            # Load scalers
            x_scaler_path = os.path.join(base_path, 'X_scaler.joblib')
            y_scaler_path = os.path.join(base_path, 'y_scaler.joblib')

            self.X_scaler = joblib.load(x_scaler_path)
            self.y_scaler = joblib.load(y_scaler_path)
        except Exception as e:
            messagebox.showerror("Model Loading Error", 
                                 f"Could not load model or scalers: {str(e)}")
            raise

    def create_ui(self):
        # Main Frame
        main_frame = tk.Frame(self.master, bg='#f5f5f5')
        main_frame.pack(padx=20, pady=20, fill='both', expand=True)

        # Title
        title_label = tk.Label(main_frame, text="Water Quality Predictor", 
                               font=("Segoe UI", 20, "bold"), bg='#f5f5f5', fg='#00796b')
        title_label.pack(pady=(0, 30))

        # Input Frame
        input_frame = tk.Frame(main_frame, bg='#f5f5f5')
        input_frame.pack(fill='x', pady=15)

        # Temperature Input
        temp_label = tk.Label(input_frame, text="Temperature (°C):", bg='#f5f5f5', fg='#00796b', font=("Segoe UI", 12))
        temp_label.pack(side='left', padx=(0, 15))
        self.temp_entry = ttk.Entry(input_frame, width=18, font=("Segoe UI", 12), style="Custom.TEntry")
        self.temp_entry.pack(side='left')

        # Date Input
        date_label = tk.Label(input_frame, text="Date (YYYY/MM):", bg='#f5f5f5', fg='#00796b', font=("Segoe UI", 12))
        date_label.pack(side='left', padx=(20, 15))
        self.date_entry = ttk.Entry(input_frame, width=18, font=("Segoe UI", 12), style="Custom.TEntry")
        self.date_entry.pack(side='left')

        # Predict Button
        predict_button = ttk.Button(main_frame, text="Predict", 
                                     command=self.predict_water_quality, style='primary.TButton')
        predict_button.pack(pady=25, ipadx=20, ipady=8)

        # Results Frame
        results_frame = tk.Frame(main_frame, bg='#f5f5f5')
        results_frame.pack(pady=15, padx=10, fill='both')

        results_title = tk.Label(results_frame, text="Prediction Results:", bg='#f5f5f5', fg='#01579b', font=("Segoe UI", 16, "bold"))
        results_title.pack(pady=10)

        # Individual Result Containers
        self.result_containers = {}
        self.result_vars = {
            'NO3': tk.StringVar(value="Nitrate (NO₃): 0.00 mg/L"),
            'SO4': tk.StringVar(value="Sulphate (SO₄): 0.00 mg/L"),
            'pH': tk.StringVar(value="pH Value: 0.00")
        }

        for param, var in self.result_vars.items():
            container = tk.Frame(results_frame, bg='#e3f2fd', bd=2, relief="groove")
            container.pack(pady=5, ipadx=10, ipady=10)
            self.result_containers[param] = container

            label = tk.Label(container, textvariable=var, bg='#e3f2fd', fg='#01579b', font=("Segoe UI", 14))
            label.pack(anchor='center', pady=5)

        # Center the results
        results_frame.pack_configure(anchor='center')

        # Custom Styles
        style = ttk.Style()
        style.theme_use('clam')

        # Entry style
        style.configure('Custom.TEntry', font=("Segoe UI", 12), padding=6, relief="flat", background="#f0f4f4", foreground="#00796b")

        # Button style
        style.configure('TButton', font=("Segoe UI", 12), background='#4CAF50', foreground='white', padding=8)
        style.map('TButton', background=[('active', '#388E3C')], foreground=[('active', 'white')])
        style.configure('primary.TButton', font=("Segoe UI", 14, "bold"), background='#00796b', foreground='white', padding=10)
        style.map('primary.TButton', background=[('active', '#004d40')], foreground=[('active', 'white')])

    def predict_water_quality(self):
        try:
            # Validate inputs
            temperature = float(self.temp_entry.get())
            date_str = self.date_entry.get()

            # Convert date string to features
            date = datetime.strptime(date_str, '%Y/%m')

            # Calculate cyclical month features
            month_sin = np.sin(2 * np.pi * date.month / 12)
            month_cos = np.cos(2 * np.pi * date.month / 12)

            # Determine base year (assuming this matches your original training)
            base_year = 2010  # Adjust this to match your original data preprocessing
            year = date.year - base_year

            # Create input array
            input_data = np.array([[temperature, month_sin, month_cos, year]])

            # Normalize input
            input_normalized = self.X_scaler.transform(input_data)

            # Make prediction
            prediction_normalized = self.model.predict(input_normalized)

            # Denormalize prediction
            prediction = self.y_scaler.inverse_transform(prediction_normalized)

            # Update result variables
            self.result_vars['NO3'].set(f"Nitrate (NO₃): {prediction[0][0]:.2f} mg/L")
            self.result_vars['SO4'].set(f"Sulphate (SO₄): {prediction[0][1]:.2f} mg/L")
            self.result_vars['pH'].set(f"pH Value: {prediction[0][2]:.2f}")

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid temperature and date.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

def main():
    root = tk.Tk()
    app = WaterQualityPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
