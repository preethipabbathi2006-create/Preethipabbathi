import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# ACTIVATION FUNCTIONS
# -----------------------------
def step_function(y):
    return 1 if y >= 0 else 0

def bipolar_step(y):
    return 1 if y >= 0 else -1

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def relu(y):
    return max(0, y)

def tanh(y):
    return np.tanh(y)

def leaky_relu(y):
    return y if y > 0 else 0.01 * y


# -----------------------------
# SUMMATION UNIT
# -----------------------------
def summation(x, weights, bias):
    return np.dot(x, weights) + bias


# -----------------------------
# ERROR (COMPARATOR UNIT)
# -----------------------------
def calculate_error(target, output):
    return (target - output) ** 2


# -----------------------------
# PERCEPTRON TRAINING FUNCTION
# -----------------------------
def train_perceptron(X, y, activation_func, lr, w, b, max_epochs=1000):
    epochs = []
    errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(X)):
            net = summation(X[i], w, b)
            output = activation_func(net)

            error = y[i] - output
            total_error += error ** 2

            # weight update
            w = w + lr * error * X[i]
            b = b + lr * error

        epochs.append(epoch)
        errors.append(total_error)

        # convergence condition
        if total_error <= 0.002:
            break

    return w, b, epochs, errors


# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__ == "__main__":

    # AND GATE DATASET (from PDF)
    X_and = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y_and = np.array([0, 0, 0, 1])

    # Initial weights (given in PDF page 2)
    w = np.array([0.2, -0.75])
    b = 10
    lr = 0.05

    # Train perceptron (STEP FUNCTION)
    final_w, final_b, epochs, errors = train_perceptron(
        X_and, y_and, step_function, lr, w, b
    )

    # OUTPUT RESULTS
    print("Final Weights:", final_w)
    print("Final Bias:", final_b)
    print("Epochs to Converge:", len(epochs))

    # Plot error vs epochs
    plt.plot(epochs, errors)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs Epochs (AND Gate)")
    plt.grid()
    plt.show()


    # -----------------------------
    # XOR DATASET
    # -----------------------------
    X_xor = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y_xor = np.array([0, 1, 1, 0])

    # Train on XOR (will NOT converge - important observation)
    final_w_xor, final_b_xor, epochs_xor, errors_xor = train_perceptron(
        X_xor, y_xor, step_function, lr, w, b
    )

    print("\nXOR Results:")
    print("Final Weights:", final_w_xor)
    print("Final Bias:", final_b_xor)
    print("Epochs:", len(epochs_xor))

    plt.plot(epochs_xor, errors_xor)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs Epochs (XOR Gate)")
    plt.grid()
    plt.show()
