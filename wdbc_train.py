
import os
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
import pickle as pkl

from models import ClusteringModel, ClassificationModel, train_test_split



def student_model_train():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    save_folder = "./models/"
    os.makedirs(save_folder, exist_ok=True)

    print("Loading the dataset")
    wdbc_data = pd.read_csv("data/wdbc.data", header=None)
    X = wdbc_data.iloc[:, 2:-1].values
    y = wdbc_data.iloc[:, 1].values

    print("Encoding target labels")
    y_int = np.zeros((len(y), 1), dtype=int)
    y_int[y == 'B'] = 0
    y_int[y == 'M'] = 1
    y = np.hstack([1 - y_int, y_int])

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training and saving the Clustering model")
    clustering_model = ClusteringModel(num_clusters=5)
    clustering_model.fit(X_train)
    with open(os.path.join(save_folder, 'clustering.pkl'), 'wb') as f:
        pkl.dump(clustering_model, f)

    print("Training and saving the Classification model")
    classif_model = ClassificationModel(input_dim=X.shape[1], output_dim=2)
    classif_model.train(X_train, y_train)
    with open(os.path.join(save_folder, 'classif.pkl'), 'wb') as f:
        pkl.dump(classif_model, f)

    print("Training and saving the Classification model with centroid descriptors")
    representation_X_train = clustering_model.compute_representation(X_train)
    classif_model_repr = ClassificationModel(input_dim=representation_X_train.shape[1], output_dim=2)
    classif_model_repr.train(representation_X_train, y_train)
    with open(os.path.join(save_folder, 'classif-centroid.pkl'), 'wb') as f:
        pkl.dump(classif_model_repr, f)

def run_all_tests():
    df = pd.read_csv("data/wdbc.data", header=None)
    df = df.drop(columns=0)
    labels = df[1].values
    y = np.array([[1, 0] if label == 'B' else [0, 1] for label in labels])
    X = df.iloc[:, 2:].values.astype(float)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    for n in [4, 8, 16, 32]:
        model = ClusteringModel(n)
        model.fit(X)
        print(f"[T{[4,8,16,32].index(n)+1}] Silhouette ({n} clusters):", model.silhouette_score(X))

    X_noisy = X + np.random.normal(0, 0.2, size=X.shape)
    model = ClusteringModel(16)
    model.fit(X_noisy)
    print("[T6] Silhouette with noise:", model.silhouette_score(X_noisy))

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2, 42)

    clf = ClassificationModel(X.shape[1], 2)
    clf.train(X_train, y_train)
    print("[T7] Accuracy on noisy data:", clf.evaluate(X_test + np.random.normal(0, 0.3, size=X_test.shape), y_test))

    model = ClusteringModel(5)
    model.fit(X_train)
    clf = ClassificationModel(X.shape[1], 2)
    clf.train(X_train, y_train)
    print("[T8] Accuracy with clustering initialized model:", clf.evaluate(X_test, y_test))
    print("[T9] Accuracy on original data:", clf.evaluate(X_test, y_test))

    preds = clf.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    print("[T10] Malignant recall:", recall_score(y_true == 1, preds == 1))
    print("[T11] Benign recall:", recall_score(y_true == 0, preds == 0))

    X_repr_train = model.compute_representation(X_train)
    X_repr_test = model.compute_representation(X_test)
    clf_repr = ClassificationModel(X_repr_train.shape[1], 2)
    clf_repr.train(X_repr_train, y_train)
    preds_repr = clf_repr.predict(X_repr_test)
    print("[T12] Malignant (repr):", recall_score(np.argmax(y_test, axis=1) == 1, preds_repr == 1))
    print("[T13] Benign (repr):", recall_score(np.argmax(y_test, axis=1) == 0, preds_repr == 0))

    print("[T14] Missing values:", clf.evaluate(np.nan_to_num(X_test, nan=0.0), y_test))
    print("[T15] Outliers:", clf.evaluate(X_test * 100, y_test))
    print("[T16] Scaled:", clf.evaluate(X_test * 2, y_test))
    print("[T17] Translated:", clf.evaluate(X_test + 1, y_test))
    print("[T18] Permuted:", clf.evaluate(X_test[:, ::-1], y_test))

    print("[T19] Outliers (repr):", clf_repr.evaluate(X_repr_test * 100, y_test))
    print("[T20] Scaled (repr):", clf_repr.evaluate(X_repr_test * 2, y_test))
    print("[T21] Translated (repr):", clf_repr.evaluate(X_repr_test + 1, y_test))

if __name__ == "__main__":
    student_model_train()
    run_all_tests()
