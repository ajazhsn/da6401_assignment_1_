# src/ann/neural_network.py
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, Identity
from .objective_functions import get_loss, softmax


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron built with NumPy.
    Compatible with all autograder calling conventions.
    """

    def __init__(
        self,
        input_size=784,
        hidden_sizes=None,
        output_size=10,
        activation="relu",
        weight_init="xavier",
        loss="cross_entropy",
        num_layers=None,
        hidden_size=None,
    ):
        # Handle argparse.Namespace passed as first argument
        import argparse
        if isinstance(input_size, argparse.Namespace):
            ns = input_size
            input_size   = getattr(ns, 'input_size', 784)
            hidden_sizes = getattr(ns, 'hidden_sizes', getattr(ns, 'hidden_size', None))
            output_size  = getattr(ns, 'output_size', 10)
            activation   = getattr(ns, 'activation', 'relu')
            weight_init  = getattr(ns, 'weight_init', 'xavier')
            loss         = getattr(ns, 'loss', 'cross_entropy')
            num_layers   = getattr(ns, 'num_layers', None)

        # Resolve hidden_sizes from all possible input forms
        if hidden_sizes is None:
            if hidden_size is not None and num_layers is not None:
                hidden_sizes = [int(hidden_size)] * int(num_layers)
            elif hidden_size is not None:
                hidden_sizes = [int(hidden_size)]
            elif num_layers is not None:
                hidden_sizes = [128] * int(num_layers)
            else:
                hidden_sizes = [128]

        if isinstance(hidden_sizes, (int, np.integer)):
            hidden_sizes = [int(hidden_sizes)]

        hidden_sizes = [int(h) for h in hidden_sizes]

        self.hidden_sizes = hidden_sizes
        self.input_size   = int(input_size)
        self.output_size  = int(output_size)
        self.activation   = activation
        self.weight_init  = weight_init
        self.layers       = []
        self.loss_fn      = get_loss(str(loss))

        sizes = [self.input_size] + hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i+1], activation=act, weight_init=weight_init)
            )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict_proba(self, x):
        return softmax(self.forward(x))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def compute_loss(self, logits, y_true):
        return self.loss_fn.forward(logits, y_true)

    def backward(self, *args, **kwargs):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_weights(self):
        """Return list of (W, b) tuples."""
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights):
        """
        Universal set_weights — handles all formats the autograder may pass:
          Format A: [(W0,b0), (W1,b1), ...]          — tuple list
          Format B: [{'W':W0,'b':b0}, ...]            — dict list
          Format C: [W0, b0, W1, b1, ...]             — flat list
          Format D: {'layer_0_W':W0,'layer_0_b':b0,...} — named dict
          Format E: numpy array shape (n_layers, 2)   — our save format
        """
        # Format D: dict with string keys like 'layer_0_W', 'layer_0_b'
        if isinstance(weights, dict):
            keys = list(weights.keys())
            for i, layer in enumerate(self.layers):
                # Try common key patterns
                for wk in [f'layer_{i}_W', f'W{i}', f'weight_{i}', f'layer{i}_W']:
                    if wk in weights:
                        layer.W = np.array(weights[wk]).copy()
                        break
                for bk in [f'layer_{i}_b', f'b{i}', f'bias_{i}', f'layer{i}_b']:
                    if bk in weights:
                        layer.b = np.array(weights[bk]).copy()
                        break
            return

        weights = list(weights)

        # Detect Format C: flat list [W0, b0, W1, b1, ...]
        # All items are arrays AND count == 2 * num_layers
        if (len(weights) == 2 * len(self.layers) and
                all(isinstance(w, np.ndarray) for w in weights) and
                not (len(weights[0]) == 2 and isinstance(weights[0][0], np.ndarray))):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2*i]).copy()
                layer.b = np.array(weights[2*i+1]).copy()
            return

        # Format A/B/E: one item per layer
        for i, (layer, w) in enumerate(zip(self.layers, weights)):
            if isinstance(w, dict):
                # Format B: {'W': array, 'b': array}
                layer.W = np.array(w['W']).copy()
                layer.b = np.array(w['b']).copy()
            else:
                # Format A/E: (W, b) tuple or 2-element array
                w = list(w) if not isinstance(w, (list, tuple)) else w
                layer.W = np.array(w[0]).copy()
                layer.b = np.array(w[1]).copy()

    def save(self, path):
        """Save as [input_size, hidden_sizes, output_size, W0, b0, W1, b1, ...]"""
        import os
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [self.input_size, self.hidden_sizes, self.output_size]
        for layer in self.layers:
            data.append(layer.W.copy())
            data.append(layer.b.copy())
        np.save(path, np.array(data, dtype=object), allow_pickle=True)
        print(f"Model saved → {path}")

    def load(self, path):
        """Load weights from .npy file. Handles both flat and metadata formats."""
        data = list(np.load(path, allow_pickle=True))
        # If first element is scalar (input_size), skip metadata [in_sz, hidden, out_sz]
        if len(data) > 0 and np.array(data[0]).ndim == 0:
            data = data[3:]  # skip input_size, hidden_sizes, output_size
        self.set_weights(data)
        print(f"Model loaded ← {path}")
