# Shallow Neural Network From Scratch

This project implements a shallow feedforward neural network **from scratch using NumPy**.
It was developed as part of an AI coursework and includes a graphical interface for interactive
experimentation with different network architectures and datasets.

The implementation avoids high-level ML frameworks and focuses on core neural network concepts
such as forward propagation, backpropagation, and gradient descent.

---

## Features

- Fully configurable neural network architecture
- Support for multiple hidden layers
- Batch-based training using gradient descent
- Sigmoid activation function
- Two-class synthetic data classification
- Full 10-class MNIST digit classification
- Graphical User Interface (GUI) for interaction and visualization

---

## Configurable Network Architecture

The network architecture is defined through the GUI using a comma-separated list.

Examples:
- `10` → one hidden layer with 10 neurons
- `16,16` → two hidden layers with 16 neurons each
- `32,16,8` → three hidden layers with 32, 16, and 8 neurons

This design allows flexible experimentation with different depths and widths.
Larger networks can model more complex patterns but may train slower,
while smaller networks train faster and work well for simpler data.

---

## Datasets

### Synthetic Data
- Two-dimensional, multi-modal Gaussian data
- Two output neurons with one-hot encoding
- Visualized decision boundaries after training

### MNIST
- Full 10-class digit classification
- Uses raw IDX files (no external ML libraries)
- Adjustable training parameters via the GUI

---

## Technologies Used

- Python
- NumPy
- Matplotlib
- Tkinter
