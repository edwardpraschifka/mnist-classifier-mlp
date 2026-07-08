# MNIST Neural Network

A feedforward neural network trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset, built from scratch using NumPy.

## Architecture

The network uses a fixed 3-layer architecture:

```
Input (784) → Hidden (128) → Hidden (64) → Output (10)
```

- **Input**: 784 nodes — one per pixel of a 28×28 grayscale image
- **Output**: 10 nodes — one per digit (0–9); the predicted digit is the node with the highest activation

Training uses mini-batch gradient descent with backpropagation.

---

## Scripts

### `src/train.py` — Train the network

Trains a new network on the MNIST training set and saves it to disk.

```bash
python -m src.train [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-b` | int | `32` | Mini-batch size. Larger values are faster per epoch but may converge to a worse minimum. |
| `-l` | float | `0.01` | Learning rate. Controls how large each gradient step is. Too high causes divergence; too low causes slow convergence. |
| `-e` | int | `30` | Number of training epochs (full passes over the training set). |
| `-a` | int | `0` | Activation function. See options below. |
| `-o` | str | `network.pkl` | Output file path where the trained network is saved. |

**Activation function options (`-a`)**

| Value | Function | Notes |
|-------|----------|-------|
| `0` | Sigmoid | Default. Squashes activations to (0, 1). |


**Example**

```bash
# Train for 50 epochs with learning rate 0.001, save to my_model.pkl
python -m src.train -e 50 -lr 0.001 -o my_model.pkl
```

---

### `src/test.py` — Evaluate the network

Loads a saved network and reports its accuracy on the MNIST test set.

```bash
python -m src.test [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f` | str | `network.pkl` | Path to the saved network file to evaluate. |

**Example**

```bash
# Evaluate a model saved to my_model.pkl
python -m src.test -f my_model.pkl
```

Output:

```
Accuracy = 97.43%
```


## Project Structure

```
src/
  train.py          # Training script
  test.py           # Evaluation script
  network.py        # Network class (feedforward, backprop, gradient descent)
  activations.py    # Activation functions (Sigmoid)
  cost_functions.py # Cost functions (CrossEntropy)
  utils.py          # Data loading and helper utilities

tests/              # Pytest test suite
network.pkl         # Default saved model (produced by train.py)
```

## Requirements

Install dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
