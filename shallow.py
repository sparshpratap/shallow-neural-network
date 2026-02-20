import numpy as np
import struct
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ======================================================
# ACTIVATIONS
# ======================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

# ======================================================
# DATA GENERATION (synthetic, multi-modal)
# ======================================================

def generate_data(samples_per_class=100):
    X0 = np.random.randn(samples_per_class, 2) + np.array([-1, -1])
    X1 = np.random.randn(samples_per_class, 2) + np.array([1, 1])

    X = np.vstack((X0, X1))
    y = np.array([0]*samples_per_class + [1]*samples_per_class)

    y_onehot = np.zeros((y.size, 2))
    y_onehot[np.arange(y.size), y] = 1

    return X, y_onehot

# ======================================================
# MNIST LOADER (FULL 10-CLASS)
# ======================================================

def load_images(filename):
    with open(filename, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols) / 255.0

def load_labels(filename):
    with open(filename, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_mnist(limit=5000):
    X = load_images("mnist/train-images.idx3-ubyte")
    y = load_labels("mnist/train-labels.idx1-ubyte")

    X = X[:limit]
    y = y[:limit]

    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1

    return X, y_onehot

# ======================================================
# SHALLOW NEURAL NETWORK (FORMULA-COMPLIANT)
# ======================================================

class ShallowNeuralNetwork:
    def __init__(self, input_dim, hidden_layers, output_dim, eta=0.1):
        self.eta = eta
        self.layers = [input_dim] + hidden_layers + [output_dim]

        self.W = []
        self.b = []

        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.1)
            self.b.append(np.zeros(self.layers[i+1]))

    def forward(self, X):
        self.A = [X]
        self.Z = []

        A = X
        for i in range(len(self.W)):
            Z = A @ self.W[i] + self.b[i]
            A = sigmoid(Z)
            self.Z.append(Z)
            self.A.append(A)

        return A

    def train(self, X, y, epochs=100, batch_size=32):
        n = X.shape[0]

        for _ in range(epochs):
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            for i in range(0, n, batch_size):
                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                y_hat = self.forward(xb)

                # ----- BACKPROP (formula-compliant) -----
                delta = (y_hat - yb) * sigmoid_derivative(y_hat)

                dW = [None]*len(self.W)
                db = [None]*len(self.b)

                for l in reversed(range(len(self.W))):
                    dW[l] = self.A[l].T @ delta / len(xb)
                    db[l] = np.mean(delta, axis=0)

                    if l > 0:
                        delta = (delta @ self.W[l].T) * sigmoid_derivative(self.A[l])

                for l in range(len(self.W)):
                    self.W[l] -= self.eta * dW[l]
                    self.b[l] -= self.eta * db[l]

    def predict(self, X):
        y = self.forward(X)
        return np.argmax(y, axis=1)

# ======================================================
# GUI APPLICATION
# ======================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Shallow Neural Network – Task 4.4")

        self.X = None
        self.y = None
        self.net = None
        self.mode = "synthetic"

        self.build_ui()
        self.build_plot()

    def build_ui(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, padx=10)

        tk.Label(frame, text="Hidden layers (e.g. 16 or 16,16)").pack()
        self.layers_entry = tk.Entry(frame)
        self.layers_entry.insert(0, "16,16")
        self.layers_entry.pack()

        tk.Button(frame, text="Generate Data", command=self.generate).pack(pady=5)
        tk.Button(frame, text="Train Synthetic", command=self.train).pack(pady=5)
        tk.Button(frame, text="Load MNIST", command=self.load_mnist).pack(pady=5)
        tk.Button(frame, text="Train MNIST", command=self.train_mnist).pack(pady=5)

        self.status = tk.Label(frame, text="Ready")
        self.status.pack(pady=10)

    def build_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

    def parse_layers(self):
        return [int(x) for x in self.layers_entry.get().split(",")]

    # ---------------- DATA ----------------

    def generate(self):
        self.mode = "synthetic"
        self.X, self.y = generate_data()

        self.net = ShallowNeuralNetwork(
            input_dim=2,
            hidden_layers=self.parse_layers(),
            output_dim=2
        )

        self.status.config(text="Synthetic data generated")
        self.plot()

    def load_mnist(self):
        self.mode = "mnist"
        self.X, self.y = load_mnist()

        self.net = ShallowNeuralNetwork(
            input_dim=784,
            hidden_layers=self.parse_layers(),
            output_dim=10
        )

        self.ax.clear()
        self.ax.text(0.5, 0.5, "MNIST loaded\n(10 classes)",
                     ha="center", va="center", transform=self.ax.transAxes)
        self.canvas.draw()

        self.status.config(text="MNIST loaded")

    # ---------------- TRAINING ----------------

    def train(self):
        self.net.train(self.X, self.y, epochs=200)
        self.status.config(text="Synthetic training complete")
        self.plot()

    def train_mnist(self):
        self.net.train(self.X, self.y, epochs=10, batch_size=64)
        preds = self.net.predict(self.X)
        acc = np.mean(preds == np.argmax(self.y, axis=1))

        self.ax.clear()
        self.ax.text(0.5, 0.5, f"MNIST Accuracy:\n{acc:.3f}",
                     ha="center", va="center", transform=self.ax.transAxes)
        self.canvas.draw()

        self.status.config(text=f"MNIST accuracy: {acc:.3f}")

    # ---------------- PLOT ----------------

    def plot(self):
        if self.mode != "synthetic":
            return

        self.ax.clear()
        labels = np.argmax(self.y, axis=1)

        self.ax.scatter(self.X[labels==0][:,0], self.X[labels==0][:,1], color="blue")
        self.ax.scatter(self.X[labels==1][:,0], self.X[labels==1][:,1], color="red")

        xx, yy = np.meshgrid(np.linspace(-4,4,200), np.linspace(-4,4,200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.net.predict(grid).reshape(xx.shape)

        self.ax.contourf(xx, yy, preds, alpha=0.25)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
