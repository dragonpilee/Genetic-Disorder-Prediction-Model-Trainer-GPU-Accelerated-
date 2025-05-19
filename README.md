# 🧬 Genetic Disorder Prediction Model Trainer (GPU-Accelerated)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/CUDA-Numba-green?logo=nvidia)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

**Genetic Disorder Prediction Model Trainer** is a modern, GPU-accelerated desktop application for training and visualizing logistic regression models on synthetic or CSV genetic disorder data. Built with Python, Tkinter, and Numba (CUDA), it provides a beautiful, interactive interface for exploring machine learning on your own GPU.


---

## ✨ Features

- **GPU-Accelerated Training:** Uses Numba and CUDA for fast logistic regression.
- **Modern GUI:** Clean, responsive interface with Tkinter and ttk themes.
- **Synthetic Data Generation:** Instantly generate data for common genetic disorders.
- **CSV Import:** Load your own datasets for custom experiments.
- **Live Visualization:** Real-time loss curve and prediction scatter plots.
- **Status Logging:** See detailed logs of training progress.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/dragonpilee/Genetic-Disorder-Prediction-Model-Trainer-GPU-Accelerated-.git
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- numpy
- pandas
- matplotlib
- numba
- tkinter (comes with standard Python on Windows)

> **Note:** You need an NVIDIA GPU with CUDA support for GPU acceleration.

### 3. Run the App

```bash
python predication_model_cuda.py
```

---

## 🖥️ Usage

- **Select Disorder:** Choose a disorder from the dropdown.
- **Generate Synthetic Data:** Click to create demo data.
- **Load CSV Data:** Import your own dataset (CSV with at least 2 features and a label column).
- **Train Model:** Start GPU-accelerated training and watch the live plots.
- **View Results:** See loss over epochs and prediction scatter plot.
- **Logs:** Check the status/log panel for progress and errors.

---

## 📊 Example CSV Format

```csv
feature1,feature2,label
0.12,-0.34,1
-0.56,0.78,0
...
```

---

## 🧑‍💻 Developed By

**Alan Cyril Sunny**  

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐️ Show Your Support

If you like this project, please ⭐️ star the repo and share it!

---

## 📝 TODO

- [ ] Add more disorders and data patterns
- [ ] Export trained model
- [ ] Add more advanced ML models
- [ ] Cross-platform packaging

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue first to discuss what you would like to change.

---

## 📬 Contact

For questions or feedback, open an issue or contact [Alan Cyril Sunny](alan_cyril@yahoo.com).
