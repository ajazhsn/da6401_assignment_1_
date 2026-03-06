# src/ann/neural_layer.py
# Defines a single fully-connected (dense) layer of the MLP.
# Each layer holds:
#   - W : weight matrix of shape (input_size, output_size)
#   - b : bias vector of shape (1, output_size)
#   - grad_W, grad_b : gradients (populated after backward())
#   - optimizer_state : dict for stateful optimizers (Adam, RMSProp, etc.)

import numpy as np
from .activations import get_activation, Identity


class NeuralLayer:
    """
    A single fully-connected layer with configurable activation.

    Forward pass:  z = x @ W + b  →  a = activation(z)
    Backward pass: computes grad_W, grad_b and returns upstream gradient
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation,
        weight_init: str = "xavier",
    ):
        """
        Args:
            input_size:   Number of input features (fan-in).
            output_size:  Number of neurons in this layer (fan-out).
            activation:   Activation object (must have forward/backward).
            weight_init:  'xavier' (recommended) or 'random' (small random).
        """
        self.activation = activation

        # Initialise weights and biases
        self._init_weights(input_size, output_size, weight_init)

        # Gradients — computed and stored during backward()
        # Autograder checks these attributes directly
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Per-layer state for stateful optimizers (momentum, Adam, etc.)
        self.optimizer_state = {}

    # ------------------------------------------------------------------
    # Weight Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self, fan_in: int, fan_out: int, method: str):
        """
        Initialise W and b.

        Xavier: keeps gradient variance stable across layers by scaling
                weights with sqrt(6 / (fan_in + fan_out)).
        Random: small Gaussian noise — breaks symmetry but less stable.
        """
        if method == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        elif method == "random":
            self.W = np.random.randn(fan_in, fan_out) * 0.01
        elif method == "zeros":
            # Used only for symmetry-breaking experiments (Q2.9)
            # WARNING: produces dead network — all neurons identical
            self.W = np.zeros((fan_in, fan_out))
        else:
            raise ValueError(
                f"Unknown weight_init '{method}'. Choose 'xavier' or 'random'."
            )
        # Biases always start at zero
        self.b = np.zeros((1, fan_out))

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute layer output.

        Args:
            x: Input array of shape (batch_size, input_size)
        Returns:
            Activated output of shape (batch_size, output_size)
        """
        self.x = x                       # cache input for backward
        self.z = x @ self.W + self.b     # linear pre-activation
        return self.activation.forward(self.z)

    # ------------------------------------------------------------------
    # Backward Pass
    # ------------------------------------------------------------------
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backpropagate gradient through this layer.
        Computes and stores self.grad_W and self.grad_b.

        Args:
            grad_output: Gradient w.r.t. layer output, shape (batch, output_size)
        Returns:
            Gradient w.r.t. layer input, shape (batch, input_size)
        """
        batch_size = self.x.shape[0]

        # Pass gradient through activation function
        grad_z = self.activation.backward(grad_output)   # (batch, output_size)

        # Gradient w.r.t. weights
        # NOTE: do NOT divide by batch_size here — the loss function
        # already returns a mean, so grad_z already accounts for batch averaging.
        self.grad_W = self.x.T @ grad_z                   # (input_size, output_size)

        # Gradient w.r.t. biases — sum over batch (consistent with grad_W)
        self.grad_b = grad_z.sum(axis=0, keepdims=True)   # (1, output_size)

        # Gradient w.r.t. input — passed to previous layer
        return grad_z @ self.W.T                          # (batch, input_size)
