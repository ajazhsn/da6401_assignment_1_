# src/ann/neural_network.py
# Defines the NeuralNetwork (MLP) class.
# Stacks NeuralLayer objects and orchestrates forward/backward passes.
# Also handles model serialisation (save/load as .npy).

import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, Identity
from .objective_functions import get_loss, softmax


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron (MLP) built entirely with NumPy.

    Architecture:
        Input → [Hidden Layers with activation] → Output (Identity, softmax in loss)

    Usage:
        model = NeuralNetwork(784, [128, 128], 10, activation='relu')
        logits = model.forward(x_batch)
        loss   = model.compute_loss(logits, y_batch)
        model.backward()
        optimizer.step(model.layers)
    """

    def __init__(
        self,
        input_size=784,
        hidden_sizes: list = None,
        output_size: int = 10,
        activation: str = "relu",
        weight_init: str = "xavier",
        loss: str = "cross_entropy",
        num_layers: int = None,
        hidden_size: int = None,
    ):
        """
        Args:
            input_size:   Number of input features, or an argparse.Namespace object.
            hidden_sizes: List of neuron counts per hidden layer (e.g. [128, 128]).
            output_size:  Number of output classes (e.g. 10 for Fashion-MNIST).
            activation:   Activation for hidden layers: 'sigmoid' | 'tanh' | 'relu'.
            weight_init:  Weight init strategy: 'xavier' | 'random'.
            loss:         Loss function: 'cross_entropy' | 'mean_squared_error'.
            num_layers:   Alternative way to specify number of hidden layers.
            hidden_size:  Alternative way to specify neurons per layer (int).
        """
        # Handle argparse.Namespace being passed as first argument
        import argparse
        if isinstance(input_size, argparse.Namespace):
            args = input_size
            input_size   = getattr(args, 'input_size', 784)
            hidden_sizes = getattr(args, 'hidden_sizes',
                          getattr(args, 'hidden_size', [128]))
            output_size  = getattr(args, 'output_size', 10)
            activation   = getattr(args, 'activation', 'relu')
            weight_init  = getattr(args, 'weight_init', 'xavier')
            loss         = getattr(args, 'loss', 'cross_entropy')
            num_layers   = getattr(args, 'num_layers', None)
            if isinstance(hidden_sizes, int):
                hidden_sizes = [hidden_sizes] * (num_layers or 1)

        # Resolve hidden_sizes from multiple possible input formats
        if hidden_sizes is None:
            if hidden_size is not None and num_layers is not None:
                # e.g. hidden_size=128, num_layers=3 → [128, 128, 128]
                hidden_sizes = [hidden_size] * num_layers
            elif hidden_size is not None:
                hidden_sizes = [hidden_size]
            elif num_layers is not None:
                hidden_sizes = [128] * num_layers
            else:
                # Sensible default
                hidden_sizes = [128]

        # If hidden_sizes is a single int, wrap it
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.hidden_sizes = hidden_sizes
        self.input_size   = input_size
        self.output_size  = output_size
        self.activation   = activation
        self.weight_init  = weight_init

        self.layers  = []
        self.loss_fn = get_loss(loss)

        # Build layer stack: input → hidden... → output
        sizes = [input_size] + list(hidden_sizes) + [output_size]

        for i in range(len(sizes) - 1):
            # Output layer uses Identity activation — loss handles softmax
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            self.layers.append(
                NeuralLayer(
                    input_size=sizes[i],
                    output_size=sizes[i + 1],
                    activation=act,
                    weight_init=weight_init,
                )
            )

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run input through all layers sequentially.

        Args:
            x: Input array of shape (batch_size, input_size)
        Returns:
            Raw logits of shape (batch_size, output_size)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out   # raw logits — softmax applied inside loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities over classes."""
        return softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class index for each sample."""
        return np.argmax(self.predict_proba(x), axis=1)

    # ------------------------------------------------------------------
    # Loss Computation
    # ------------------------------------------------------------------
    def compute_loss(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute scalar loss and cache state for backward pass.

        Args:
            logits: Raw network output (batch, output_size)
            y_true: Integer class labels (batch,)
        Returns:
            Scalar loss value
        """
        return self.loss_fn.forward(logits, y_true)

    # ------------------------------------------------------------------
    # Backward Pass
    # ------------------------------------------------------------------
    def backward(self):
        """
        Backpropagate gradients from loss through all layers in reverse.
        After this call, each layer's .grad_W and .grad_b are populated
        and ready for the optimizer to use.
        """
        grad = self.loss_fn.backward()          # gradient w.r.t. output logits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)         # propagate backward layer by layer

    # ------------------------------------------------------------------
    # Model Serialisation
    # ------------------------------------------------------------------
    def get_weights(self) -> list:
        """Return all layer weights as list of (W, b) tuples."""
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights: list):
        """Load weights from list of (W, b) tuples."""
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = W.copy()
            layer.b = b.copy()

    def save(self, path: str):
        """
        Serialize all layer weights to a .npy file.
        Uses allow_pickle=True to store list of arrays.
        """
        weights = self.get_weights()
        np.save(path, np.array(weights, dtype=object), allow_pickle=True)
        print(f"Model saved → {path}")

    def load(self, path: str):
        """Load layer weights from a previously saved .npy file."""
        weights = np.load(path, allow_pickle=True)
        self.set_weights(list(weights))
        print(f"Model loaded ← {path}")
