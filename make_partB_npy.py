import numpy as np

def make_quickdraw(in_path: str, out_path: str, max_n: int = 20000):
    arr = np.load(in_path)  # (N, 784), uint8
    if max_n is not None:
        arr = arr[:max_n]
    arr = arr.reshape(-1, 28, 28).astype(np.float32) / 255.0   # (N,28,28) in [0,1]
    arr = arr[:, None, :, :]                                   # (N,1,28,28)
    np.save(out_path, arr)
    print("Saved QuickDraw:", out_path, arr.shape, arr.dtype, arr.min(), arr.max())

def make_mnist(out_path_train: str, out_path_test: str):
    # Uses sklearn’s built-in MNIST fetch (no torchvision needed).
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    X = X.reshape(-1, 28, 28)          # (70000,28,28)

    # Standard split: first 60k train, last 10k test
    X_train = X[:60000][:, None, :, :]  # (60000,1,28,28)
    X_test  = X[60000:][:, None, :, :]  # (10000,1,28,28)

    np.save(out_path_train, X_train)
    np.save(out_path_test, X_test)
    print("Saved MNIST train:", out_path_train, X_train.shape)
    print("Saved MNIST test:", out_path_test, X_test.shape)

if __name__ == "__main__":
    make_mnist("data/partB/mnist_train.npy", "data/partB/mnist_test.npy")
    make_quickdraw("data/partB/quickdraw_cat.npy", "data/partB/quickdraw_test.npy", max_n=10000)

