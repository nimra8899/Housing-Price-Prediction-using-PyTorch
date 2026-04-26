import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

# ================= MODEL =================
class SimpleRegressionNet(nn.Module):
    def __init__(self):
        super(SimpleRegressionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleRegressionNet()
scaler = None  # IMPORTANT

# ================= DATASET =================
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= GUI =================
root = tk.Tk()
root.title("Housing Price Regression GUI")
root.geometry("1000x700")

# ================= LOAD MODEL =================
def load_model():
    global scaler

    file_path = filedialog.askopenfilename()
    checkpoint = torch.load(file_path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']

    model.eval()
    status_label.config(text="Model Loaded Successfully ✅")

# ================= RUN INFERENCE (CLICK) =================
def run_selected_inference(event):
    if scaler is None:
        status_label.config(text="⚠️ Load model first!")
        return

    if not event.widget.curselection():
        return

    index = event.widget.curselection()[0]
    selection = event.widget.get(index)

    # DEBUG (optional)
    print("Clicked:", selection)

    values = list(map(float, selection.split(',')))

    features = np.array(values[:8]).reshape(1, -1)
    label = values[8]

    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(features)

    pred_label.config(text=f"Predicted: {prediction.item():.4f}")
    true_label.config(text=f"Ground Truth: {label:.4f}")

# ================= MANUAL INFERENCE =================
def manual_inference():
    if scaler is None:
        status_label.config(text="⚠️ Load model first!")
        return

    try:
        values = [float(entry.get()) for entry in entries]
        features = np.array(values).reshape(1, -1)

        features = scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(features)

        pred_label.config(text=f"Predicted: {prediction.item():.4f}")

    except:
        pred_label.config(text="Invalid Input ❌")

# ================= BUTTONS =================
tk.Button(root, text="Load Model", command=load_model, bg="blue", fg="white").pack(pady=10)

# ================= LISTBOXES =================
frame = tk.Frame(root)
frame.pack()

train_list = tk.Listbox(frame, width=60, height=20)
train_list.pack(side="left", padx=10)

val_list = tk.Listbox(frame, width=60, height=20)
val_list.pack(side="right", padx=10)

# ================= FILL DATA (LIMITED) =================
for i in range(200):  # limit for better UI
    row = list(X_train[i]) + [y_train[i][0]]
    train_list.insert(tk.END, ",".join(map(str, row)))

for i in range(200):
    row = list(X_val[i]) + [y_val[i][0]]
    val_list.insert(tk.END, ",".join(map(str, row)))

train_list.bind('<<ListboxSelect>>', run_selected_inference)
val_list.bind('<<ListboxSelect>>', run_selected_inference)

# ================= MANUAL INPUT =================
labels = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
          "Population", "AveOccup", "Latitude", "Longitude"]

entries = []

for label_text in labels:
    frame = tk.Frame(root)
    frame.pack(pady=2)

    tk.Label(frame, text=label_text, width=15).pack(side="left")
    entry = tk.Entry(frame)
    entry.pack(side="left")

    entries.append(entry)

tk.Button(root, text="Run Manual Inference", command=manual_inference, bg="green", fg="white").pack(pady=10)

# ================= OUTPUT =================
pred_label = tk.Label(root, text="Predicted: ", font=("Arial", 14))
pred_label.pack()

true_label = tk.Label(root, text="Ground Truth: ", font=("Arial", 14))
true_label.pack()

status_label = tk.Label(root, text="", fg="red")
status_label.pack()

root.mainloop()