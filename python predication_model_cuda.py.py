import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numba import cuda
import math

# ========== GPU Logistic Regression Helpers ==========

@cuda.jit(device=True)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

@cuda.jit
def predict_kernel(X, weights, y_pred):
    i = cuda.grid(1)
    if i < X.shape[0]:
        s = 0.0
        for j in range(X.shape[1]):
            s += X[i, j] * weights[j]
        y_pred[i] = sigmoid(s)

@cuda.jit
def gradient_kernel(X, y, y_pred, grad):
    i = cuda.grid(1)
    if i < X.shape[0]:
        error = y[i] - y_pred[i]
        for j in range(X.shape[1]):
            cuda.atomic.add(grad, j, error * X[i, j])

def train_logistic_regression(X, y, weights, lr, epochs, update_callback=None):
    threads_per_block = 32
    blocks_per_grid = (X.shape[0] + (threads_per_block - 1)) // threads_per_block

    d_X = cuda.to_device(X)
    d_y = cuda.to_device(y)
    d_weights = cuda.to_device(weights)
    d_y_pred = cuda.device_array(X.shape[0], dtype=np.float32)
    d_grad = cuda.device_array(weights.shape[0], dtype=np.float32)

    losses = []

    for epoch in range(epochs):
        d_y_pred[:] = 0
        d_grad[:] = 0

        predict_kernel[blocks_per_grid, threads_per_block](d_X, d_weights, d_y_pred)
        cuda.synchronize()

        y_pred = d_y_pred.copy_to_host()
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        losses.append(loss)

        d_grad[:] = 0
        gradient_kernel[blocks_per_grid, threads_per_block](d_X, d_y, d_y_pred, d_grad)
        cuda.synchronize()

        grad = d_grad.copy_to_host()
        grad /= X.shape[0]

        weights += lr * grad
        d_weights = cuda.to_device(weights)

        if update_callback:
            update_callback(epoch + 1, loss, weights, y_pred)

    return weights, losses, y_pred

# ========== Data Generation for Disorders ==========

def generate_data_for_disorder(disorder_name, samples=500):
    np.random.seed(42)
    if disorder_name == "Asthma":
        X = np.random.randn(samples, 2)
        y = (X[:, 0] + X[:, 1] > 0.5).astype(np.float32)
    elif disorder_name == "Epilepsy":
        X = np.random.randn(samples, 2)
        y = (X[:, 0] * X[:, 1] > 0).astype(np.float32)
    elif disorder_name == "Cystic Fibrosis":
        X = np.random.randn(samples, 2)
        y = ((X[:, 0]**2 + X[:, 1]**2) < 1).astype(np.float32)
    elif disorder_name == "Huntington's Disease":
        X = np.random.randn(samples, 2)
        y = (X[:, 0] - X[:, 1] > 0).astype(np.float32)
    else:
        X = np.random.randn(samples, 2)
        y = (np.random.rand(samples) > 0.5).astype(np.float32)
    return X.astype(np.float32), y

# ========== Main App Class ==========

class GeneticDisorderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Genetic Disorder Prediction Model Trainer")
        self.geometry("1920x1080")
        self.configure(bg="white")

        # GPU info
        self.gpu_info = self.get_gpu_info()

        # Data & model placeholders
        self.data = None
        self.labels = None
        self.weights = None
        self.losses = []
        self.predictions = None

        self.create_widgets()

    def get_gpu_info(self):
        try:
            device = cuda.get_current_device()
            name = device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name
            version = f"{device.compute_capability[0]}.{device.compute_capability[1]}"
            return f"CUDA Device: {name} (Compute Capability {version})"
        except Exception:
            return "No CUDA-compatible GPU detected"

    def create_widgets(self):
        # Use ttk styles for a modern look
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", font=("Segoe UI", 11), background="white")
        style.configure("TFrame", background="white")
        style.configure("TCombobox", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#2e3f4f", background="white")
        style.configure("Section.TLabelframe", font=("Segoe UI", 12, "bold"), background="#f5f7fa")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 12, "bold"), foreground="#2e3f4f")

        # Title
        title = ttk.Label(self, text="Genetic Disorder Prediction Model Trainer", style="Header.TLabel")
        title.pack(pady=(10, 2))

        # GPU Info Label
        self.gpu_label = ttk.Label(self, text=self.gpu_info, font=("Segoe UI", 10), background="white", foreground="#00796b")
        self.gpu_label.pack(pady=(0, 10))

        # Controls Frame
        controls = ttk.Frame(self)
        controls.pack(pady=5, padx=10, fill="x")

        ttk.Label(controls, text="Select Disorder:").pack(side="left", padx=5)
        disorders = ["Asthma", "Epilepsy", "Cystic Fibrosis", "Huntington's Disease"]
        self.disorder_var = tk.StringVar(value=disorders[0])
        disorder_menu = ttk.Combobox(controls, textvariable=self.disorder_var, values=disorders, state="readonly", width=22)
        disorder_menu.pack(side="left", padx=5)

        self.generate_btn = ttk.Button(controls, text="Generate Synthetic Data", command=self.generate_data)
        self.generate_btn.pack(side="left", padx=8)

        self.load_btn = ttk.Button(controls, text="Load CSV Data", command=self.load_csv)
        self.load_btn.pack(side="left", padx=8)

        self.train_btn = ttk.Button(controls, text="Train Model", command=self.start_training_thread)
        self.train_btn.pack(side="left", padx=8)

        # Data display
        data_frame = ttk.Labelframe(self, text="Data Preview", style="Section.TLabelframe")
        data_frame.pack(pady=10, padx=10, fill="x")
        self.data_text = tk.Text(data_frame, height=10, wrap="none", font=("Consolas", 10), bg="#f5f7fa")
        self.data_text.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        data_scroll = ttk.Scrollbar(data_frame, command=self.data_text.yview)
        data_scroll.pack(side="right", fill="y", pady=5)
        self.data_text.config(yscrollcommand=data_scroll.set)

        # Prediction & Loss plot
        plot_frame = ttk.Labelframe(self, text="Training Progress", style="Section.TLabelframe")
        plot_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # GPU processing log
        log_frame = ttk.Labelframe(self, text="Status / Log", style="Section.TLabelframe")
        log_frame.pack(fill="x", padx=10, pady=10)
        self.log_text = tk.Text(log_frame, height=6, bg="#23272e", fg="#00ff99", font=("Consolas", 10))
        self.log_text.pack(fill="x", padx=5, pady=5)
        self.log_text.insert(tk.END, "Ready.\n")

        # Developed by label with separator (move to top right)
        top_frame = ttk.Frame(self)
        top_frame.place(relx=1.0, y=0, anchor="ne")  # Top right corner

        dev_label = ttk.Label(
            top_frame,
            text="Developed by Alan Cyril Sunny",
            font=("Segoe UI", 12, "italic"),
            foreground="#00796b",
            background="white",
            anchor="e"
        )
        dev_label.pack(padx=20, pady=10, anchor="e")

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def generate_data(self):
        disorder = self.disorder_var.get()
        self.log(f"Generating synthetic data for {disorder}...")
        X, y = generate_data_for_disorder(disorder, samples=500)
        self.data = X
        self.labels = y
        self.display_data()
        self.log("Data generated.")

    def load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
        try:
            df = pd.read_csv(filepath)
            if df.shape[1] < 3:
                messagebox.showerror("Error", "CSV must have at least 3 columns: features + label")
                return
            self.data = df.iloc[:, :-1].values.astype(np.float32)
            self.labels = df.iloc[:, -1].values.astype(np.float32)
            self.display_data()
            self.log(f"Loaded data from {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def display_data(self):
        self.data_text.delete("1.0", tk.END)
        if self.data is None or self.labels is None:
            return
        header = "Feature1\tFeature2\tLabel\n"
        self.data_text.insert(tk.END, header)
        for i in range(min(len(self.labels), 100)):  # show max 100 rows
            line = f"{self.data[i,0]:.3f}\t{self.data[i,1]:.3f}\t{int(self.labels[i])}\n"
            self.data_text.insert(tk.END, line)

    def start_training_thread(self):
        if self.data is None or self.labels is None:
            messagebox.showwarning("No Data", "Please generate or load data before training.")
            return
        self.train_btn.config(state="disabled")
        self.thread = threading.Thread(target=self.train_model)
        self.thread.start()

    def train_model(self):
        lr = 0.1
        epochs = 50
        n_features = self.data.shape[1]
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.losses = []

        def update_callback(epoch, loss, weights, preds):
            self.losses.append(loss)
            self.predictions = preds
            self.log(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
            self.update_plots()
            self.update()

        self.log("Starting training on GPU...")
        weights, losses, preds = train_logistic_regression(self.data, self.labels, self.weights, lr, epochs, update_callback)
        self.weights = weights
        self.log("Training complete.")
        self.train_btn.config(state="normal")

    def update_plots(self):
        self.axs[0].clear()
        self.axs[1].clear()

        # Plot loss
        self.axs[0].set_title("Loss over epochs")
        self.axs[0].set_xlabel("Epoch")
        self.axs[0].set_ylabel("Loss")
        self.axs[0].plot(self.losses, 'r-')

        # Plot prediction scatter with true labels
        if self.data is not None and self.predictions is not None:
            self.axs[1].set_title("Predictions (red=class1, blue=class0)")
            self.axs[1].scatter(self.data[:, 0], self.data[:, 1], c=self.predictions > 0.5, cmap="bwr", alpha=0.6)

        self.canvas.draw()

if __name__ == "__main__":
    app = GeneticDisorderApp()
    app.mainloop()
