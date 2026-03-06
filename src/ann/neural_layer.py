# src/ann/neural_layer.py
import numpy as np
from .activations import get_activation, Identity


class NeuralLayer:
    """
    Single fully-connected layer.
    Forward:  z = x @ W + b  →  a = activation(z)
    Backward: stores grad_W, grad_b; returns upstream gradient
    """

    def __init__(self, input_size, output_size, activation, weight_init="xavier"):
        self.activation = activation
        self._init_weights(input_size, output_size, weight_init)
        # Autograder checks these directly after backward()
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.optimizer_state = {}

    def _init_weights(self, fan_in, fan_out, method):
        if method == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        elif method == "random":
            self.W = np.random.randn(fan_in, fan_out) * 0.01
        elif method == "zeros":
            self.W = np.zeros((fan_in, fan_out))
        else:
            raise ValueError(f"Unknown weight_init '{method}'")
        self.b = np.zeros((1, fan_out))

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return self.activation.forward(self.z)

    def backward(self, grad_output):
        """
        Backprop through this layer.
        IMPORTANT: grad_W = x.T @ grad_z (NO division by batch_size)
        The loss already returns a mean, so grad_z is already batch-averaged.
        Dividing again would give wrong gradients (off by factor of batch_size).
        """
        grad_z = self.activation.backward(grad_output)
        self.grad_W = self.x.T @ grad_z          # (input_size, output_size)
        self.grad_b = grad_z.sum(axis=0, keepdims=True)  # (1, output_size)
        return grad_z @ self.W.T                  # upstream gradient
