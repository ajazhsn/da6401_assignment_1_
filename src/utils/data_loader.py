# src/utils/data_loader.py
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_name: str, val_split: float = 0.1):
    """
    Load, flatten, and normalise MNIST / Fashion-MNIST.
    Returns: x_train, x_val, x_test, y_train, y_val, y_test
    """
    def _load_keras(name):
        try:
            from tensorflow.keras.datasets import mnist, fashion_mnist
            if name == 'mnist': return mnist.load_data()
            return fashion_mnist.load_data()
        except Exception:
            pass
        try:
            from keras.datasets import mnist, fashion_mnist
            if name == 'mnist': return mnist.load_data()
            return fashion_mnist.load_data()
        except Exception:
            pass
        # Final fallback: download manually
        import urllib.request, gzip, os, tempfile
        base = 'http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/' if 'fashion' in name else 'http://yann.lecun.com/exdb/mnist/'
        files = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        data = []
        tmpdir = tempfile.mkdtemp()
        for fname in files:
            path = os.path.join(tmpdir, fname)
            urllib.request.urlretrieve(base + fname, path)
            with gzip.open(path, 'rb') as f:
                raw = np.frombuffer(f.read(), np.uint8)
            if 'images' in fname:
                data.append(raw[16:].reshape(-1, 28, 28))
            else:
                data.append(raw[8:])
        return (data[0], data[1]), (data[2], data[3])

    name = dataset_name.lower().replace('-', '_')
    if name not in ('mnist', 'fashion_mnist'):
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    (x_train_full, y_train_full), (x_test, y_test) = _load_keras(name)

    x_train_full = x_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    x_test       = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_split, random_state=42, stratify=y_train_full,
    )

    print(f"Dataset: {dataset_name} | Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    return x_train, x_val, x_test, y_train, y_val, y_test
