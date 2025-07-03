import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# === Part 1 : ClusteringModel (KMeans et silhouette_score) ===
class ClusteringModel:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.labels = []
        self.centroids = None

    def fit(self, data, max_iter=300):
        np.random.seed(0)
        n_samples = data.shape[0]
        self.centroids = data[np.random.choice(n_samples, self.num_clusters, replace=False)]

        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([data[self.labels == i].mean(axis=0) if np.any(self.labels == i) else self.centroids[i] for i in range(self.num_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def silhouette_score(self, data):
        if self.centroids is None or len(self.labels) == 0:
            raise ValueError("Fit the model before computing silhouette score.")
        scores = []
        for i in range(len(data)):
            same_cluster = data[self.labels == self.labels[i]]
            other_clusters = [data[self.labels == j] for j in range(self.num_clusters) if j != self.labels[i]]
            a = np.mean(np.linalg.norm(same_cluster - data[i], axis=1)) if len(same_cluster) > 1 else 0
            b = np.min([np.mean(np.linalg.norm(cluster - data[i], axis=1)) for cluster in other_clusters])
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            scores.append(s)
        return np.mean(scores)

    def compute_representation(self, X):
        if self.centroids is None:
            raise ValueError("Model must be fitted before computing representation.")
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)


# === Partie 2 : ClassificationModel (Softmax Regression) ===
class ClassificationModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def _softmax(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def train(self, X_train, y_train, lr=0.1, epochs=1000):
        m = X_train.shape[0]
        for _ in range(epochs):
            Z = np.dot(X_train, self.W) + self.b
            A = self._softmax(Z)
            dZ = A - y_train
            dW = np.dot(X_train.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.W -= lr * dW
            self.b -= lr * db

    def predict(self, X_test):
        Z = np.dot(X_test, self.W) + self.b
        A = self._softmax(Z)
        return np.argmax(A, axis=1)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        return {
            "precision": precision_score(y_true, y_pred, average='binary'),
            "recall": recall_score(y_true, y_pred, average='binary'),
            "f1": f1_score(y_true, y_pred, average='binary')
        }


# === train/test split ===
def train_test_split(X, y, test_size=0.2, seed=None):
    if seed:
        np.random.seed(seed)
    indices = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0] * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]



if __name__ == "__main__":

    df = pd.read_csv("wdbc.data", header=None)
    df = df.drop(columns=0)
    labels = df[1].values
    y = np.array([[1, 0] if label == 'B' else [0, 1] for label in labels])
    X = df.iloc[:, 2:].values.astype(float)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # === T1 à T5 : silhouette avec différents clusters ===
    for n in [4, 8, 16, 32]:
        model = ClusteringModel(n)
        model.fit(X)
        print(f"[Q{[4,8,16,32].index(n)+1}] Silhouette ({n} clusters):", model.silhouette_score(X))

    # === T6 : silhouette avec bruit ===
    X_noisy = X + np.random.normal(0, 0.2, size=X.shape)
    model = ClusteringModel(16)
    model.fit(X_noisy)
    print("[Q6] Silhouette with noise:", model.silhouette_score(X_noisy))

    # === Split pour les autres tests ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2, 42)

    # === T7 : Classification brute ===
    clf = ClassificationModel(X.shape[1], 2)
    clf.train(X_train, y_train)
    print("[T7] Accuracy on noisy data:", clf.evaluate(X_test + np.random.normal(0, 0.3, size=X_test.shape), y_test))

    # === T8 : Classification initialisée avec clustering ===
    model = ClusteringModel(5)
    model.fit(X_train)
    clf = ClassificationModel(X.shape[1], 2)
    clf.train(X_train, y_train)
    print("[T8] Accuracy with clustering initialized model:", clf.evaluate(X_test, y_test))

    # === T9 : Classification sur données originales ===
    print("[T9] Accuracy on original data:", clf.evaluate(X_test, y_test))

    # === T10 & T11 : Tous les malins / bénins ===
    preds = clf.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    print("[T10] Malignant recall:", recall_score(y_true == 1, preds == 1))
    print("[T11] Benign recall:", recall_score(y_true == 0, preds == 0))

    # === T12 & T13 : représentation centroïdes ===
    X_repr_train = model.compute_representation(X_train)
    X_repr_test = model.compute_representation(X_test)
    clf_repr = ClassificationModel(X_repr_train.shape[1], 2)
    clf_repr.train(X_repr_train, y_train)
    preds_repr = clf_repr.predict(X_repr_test)
    print("[T12] Malignant (repr):", recall_score(np.argmax(y_test, axis=1) == 1, preds_repr == 1))
    print("[T13] Benign (repr):", recall_score(np.argmax(y_test, axis=1) == 0, preds_repr == 0))

    # === T14 à T17 : données transformées ===
    print("[T14] Missing values:", clf.evaluate(np.nan_to_num(X_test, nan=0.0), y_test))
    print("[T15] Outliers:", clf.evaluate(X_test * 100, y_test))
    print("[T16] Scaled:", clf.evaluate(X_test * 2, y_test))
    print("[T17] Translated:", clf.evaluate(X_test + 1, y_test))
    print("[T18] Permuted:", clf.evaluate(X_test[:, ::-1], y_test))

    # === T19 à T21 : version clustering representation ===
    print("[T19] Outliers (repr):", clf_repr.evaluate(X_repr_test * 100, y_test))
    print("[T20] Scaled (repr):", clf_repr.evaluate(X_repr_test * 2, y_test))
    print("[T21] Translated (repr):", clf_repr.evaluate(X_repr_test + 1, y_test))

# Note: Ensure that the dataset "wdbc.data" is in the same directory as this script or provide the correct path.





