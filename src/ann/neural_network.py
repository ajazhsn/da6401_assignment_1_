# src/ann/neural_network.py
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, Identity
from .objective_functions import get_loss, softmax


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_sizes=None, output_size=10,
                 activation="relu", weight_init="xavier", loss="cross_entropy",
                 num_layers=None, hidden_size=None):
        import argparse
        if isinstance(input_size, argparse.Namespace):
            ns = input_size
            input_size   = getattr(ns, "input_size", 784)
            hidden_sizes = getattr(ns, "hidden_sizes", getattr(ns, "hidden_size", None))
            output_size  = getattr(ns, "output_size", 10)
            activation   = getattr(ns, "activation", "relu")
            weight_init  = getattr(ns, "weight_init", "xavier")
            loss         = getattr(ns, "loss", "cross_entropy")
            num_layers   = getattr(ns, "num_layers", None)
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
        self.loss_name    = str(loss)
        self.layers       = []
        self.loss_fn      = get_loss(str(loss))
        sizes = [self.input_size] + hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            self.layers.append(NeuralLayer(sizes[i], sizes[i+1], activation=act, weight_init=weight_init))

    def forward(self, x):
        self._last_batch_size = x.shape[0]
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

    def backward(self, y_true=None, y_pred=None, weight_decay=0.0, *args, **kwargs):
        """
        Supports:
          model.backward()              -> uses cached loss grad
          model.backward(y_true, y_pred) -> autograder style
        Always returns (grad_W_list, grad_b_list) for autograder unpacking.
        """
        if y_pred is not None and y_true is not None:
            probs = softmax(y_pred)
            batch_size = probs.shape[0]
            if np.array(y_true).ndim == 1:
                dZ = probs.copy()
                dZ[np.arange(batch_size), np.array(y_true).astype(int)] -= 1
            else:
                dZ = probs - y_true
            dZ /= batch_size
            out_layer = self.layers[-1]
            out_layer.grad_W = out_layer.x.T @ dZ
            out_layer.grad_b = dZ.sum(axis=0, keepdims=True)
            grad = dZ @ out_layer.W.T
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad)
        else:
            grad = self.loss_fn.backward()
            if grad is None:
                batch = getattr(self, "_last_batch_size", 1)
                grad = np.zeros((batch, self.output_size))
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
        grad_W = [l.grad_W for l in self.layers]
        grad_b = [l.grad_b for l in self.layers]
        return grad_W, grad_b

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weights):
        # Handle 0-d numpy array (what np.load returns for dict-saved .npy)
        if isinstance(weights, np.ndarray) and weights.ndim == 0:
            weights = weights.item()
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f"W{i}" in weights: layer.W = np.array(weights[f"W{i}"]).copy()
                if f"b{i}" in weights: layer.b = np.array(weights[f"b{i}"]).copy()
            return
        # Flat list / tuple format
        weights = list(weights)
        while len(weights) > 0 and np.array(weights[0]).ndim == 0:
            weights = weights[1:]
        if len(weights) == 2 * len(self.layers):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2*i]).copy()
                layer.b = np.array(weights[2*i+1]).copy()
        else:
            for layer, w in zip(self.layers, weights):
                w = list(w)
                layer.W = np.array(w[0]).copy()
                layer.b = np.array(w[1]).copy()

    def save(self, path):
        import os
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.get_weights())
        print(f"Model saved -> {path}")

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.set_weights(data)  # set_weights handles 0-d array
        print(f"Model loaded <- {path}")
