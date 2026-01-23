import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from scipy.spatial.distance import minkowski

def generate_two_class_dataset(n_per_class=60, n_features=2, seed=42):
    np.random.seed(seed)

    class0 = np.random.normal(loc=2.0, scale=1.0, size=(n_per_class, n_features))

    class1 = np.random.normal(loc=6.0, scale=1.0, size=(n_per_class, n_features))

    x = np.vstack((class0, class1))
    y = np.array([0] * n_per_class + [1] * n_per_class)

    return x, y

#A1
def dot_product_custom(vector_a, vector_b):
    vector_a = np.array(vector_a, dtype=float)
    vector_b = np.array(vector_b, dtype=float)
    return float(np.sum(vector_a * vector_b))

def euclidean_norm_custom(vector_a):
    vector_a = np.array(vector_a, dtype=float)
    return float(np.sqrt(np.sum(vector_a ** 2)))
#A2

def mean_custom(values):
    values = np.array(values, dtype=float)
    return  float(np.sum(values) / len(values))

def variance_custom(values):
    values = np.array(values, dtype=float)
    mu = mean_custom(values)
    return float(np.sum((values - mu) ** 2) / len(values))

def std_custom(values):
    return float(np.sqrt(variance_custom(values)))

def class_centroid(feature_matrix):
    feature_matrix = np.array(feature_matrix, dtype=float)
    return np.mean(feature_matrix, axis=0)

def class_spread(feature_matrix):
    feature_matrix = np.array(feature_matrix, dtype=float)
    return np.std(feature_matrix, axis=0)

def interclass_distance(centroid1, centroid2):
    return float(np.linalg.norm(centroid1 - centroid2))

#A3
def feature_histogram_data(feature_vector, bins=10):
    feature_vector = np.array(feature_vector, dtype=float)
    hist_values, bin_edges = np.histogram(feature_vector, bins=bins)
    return hist_values, bin_edges

def feature_mean_variance(feature_vector):
    feature_vector = np.array(feature_vector, dtype=float)
    mu = np.mean(feature_vector)
    var = np.var(feature_vector)
    return float(mu), float(var)

#A4

def minkowski_distance_custom(vector_a, vector_b, p):
    vector_a = np.array(vector_a, dtype=float)
    vector_b = np.array(vector_b, dtype=float)
    return float(np.sum(np.abs(vector_a - vector_b) ** p) ** (1 / p))

def minkowski_distances_p1_to_10(vector_a, vector_b):
    distances = []
    for p in range(1, 11):
            distances.append(minkowski_distance_custom(vector_a, vector_b, p))
    return distances

#A5

def compare_minkowski_custom_vs_scipy(vector_a, vector_b, p):
    d_custom = minkowski_distance_custom(vector_a, vector_b, p)
    d_scipy = float(minkowski(vector_a, vector_b, p))
    return d_custom, d_scipy

#A7-A9

def train_knn_sklearn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train , y_train)
    return model

def knn_accuracy_sklearn(model, X_test, y_test):
    return float(model.score(X_test, y_test))

def knn_predictions_sklearn(model, X_test):
    return model.predict(X_test)

#A10


def euclidean_distance_custom(vector_a, vector_b):
    vector_a = np.array(vector_a, dtype=float)
    vector_b = np.array(vector_b, dtype=float)
    return float(np.sqrt(np.sum((vector_a - vector_b) ** 2)))

def knn_predict_custom(X_train, y_train, test_vector, k=3):
    distances = []
    for i in range(len(X_train)):
        d = euclidean_distance_custom(X_train[i], test_vector)
        distances.append((d, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]

    labels = [label for _, label in k_neighbors]
    unique_labels, counts = np.unique(labels, return_counts=True)

    return unique_labels[np.argmax(counts)]

def knn_predict_all_custom(X_train, y_train, X_test, k=3):
    predictions = []
    for test_vector in X_test:
        predictions.append(knn_predict_custom(X_train, y_train, test_vector, k))
    return np.array(predictions)

def accuracy_custom(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sum(y_true == y_pred) / len(y_true))

#A11

def accuracy_for_k_values_sklearn(X_train, y_train, X_test, y_test, k_min=1, k_max=11):
    max_possible_k = len(X_train)
    k_max = min(k_max, max_possible_k)

    k_values = []
    acc_values = []

    for k in range(k_min, k_max + 1):
        model = train_knn_sklearn(X_train, y_train, k=k)
        acc = knn_accuracy_sklearn(model, X_test, y_test)
        k_values.append(k)
        acc_values.append(acc)

    return k_values, acc_values

#A12

def precision_recall_f1_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp +fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall)  / (precision + recall) if (precision + recall) != 0 else 0

    return float(accuracy), float(precision), float(recall), float(f1)

#A13

def confusion_matrix_custom(y_true, y_pred, positive_label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    tn = np.sum((y_true != positive_label) & (y_pred != positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))

    cm = np.array([[tn, fp],
                   [fn, tp]])
    return cm

def metrics_from_cm_custom(cm, beta=1.0):
    tn, fp ,fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    if precision == 0 and recall == 0:
        f_beta = 0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return float(accuracy), float(precision), float(recall), float(f_beta)

#A14

def least_squares_classifier_train(X_train, y_train):
    y_mapped = np.where(y_train == 0, -1, 1).reshape(-1, 1)

    X_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

    w = np.linalg.pinv(X_aug) @ y_mapped
    return w

def least_squares_classifier_predict(X_test, w):
    X_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    scores = X_aug @ w
    preds = np.where(scores.flatten() >= 0, 1, 0)
    return preds

#main program

def main():
    X, y = generate_two_class_dataset(n_per_class=60, n_features=2, seed=42)
#A1
    vector_a = X[0]
    vector_b = X[1]

    dp_custom = dot_product_custom(vector_a, vector_b)
    dp_numpy = float(np.dot(vector_a, vector_b))

    norm_custom = euclidean_norm_custom(vector_a)
    norm_numpy = float(np.linalg.norm(vector_a))

    print("\nA1 Results:")
    print("Custom Dot Product:", dp_custom)
    print("Numpy Dot Product :", dp_numpy)
    print("Custom Norm       :", norm_custom)
    print("Numpy Norm        :", norm_numpy)
#A2
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]

    centroid0 = class_centroid(X_class0)
    centroid1 = class_centroid(X_class1)

    spread0 = class_spread(X_class0)
    spread1 = class_spread(X_class1)

    inter_dist = interclass_distance(centroid0, centroid1)

    print("\nA2 Results:")
    print("Centroid Class 0:", centroid0)
    print("centroid Class 1:", centroid1)
    print("spread (std) Class0:", spread0)
    print("spread (std) Class1:", spread1)
    print("Interclass Distance:", inter_dist)

#A3

    feature_index = 0
    feature_vec = X[:, feature_index]

    hist_vals, bin_edges = feature_histogram_data(feature_vec, bins=8)
    mu, var = feature_mean_variance(feature_vec)

    print("\nA3 Results:")
    print("Mean:", mu)
    print("Variance:", var)

    plt.figure()
    plt.hist(feature_vec, bins=8)
    plt.title("A3 Histogram for Feature " + str(feature_index))
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.show()
#A4
    distances_p = minkowski_distances_p1_to_10(vector_a, vector_b)

    print("\nA4 Results (Minkowski p=1...10):")
    for p in range(1, 11):
        print("p =", p, "distance =", distances_p[p-1])

    plt.figure()
    plt.plot(range(1, 11), distances_p, marker="o")
    plt.title("A4 Minkowski Distance vs p")
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

#A5
    p_value = 3
    d_custom, d_scipy = compare_minkowski_custom_vs_scipy(vector_a, vector_b, p_value)

    print("\nA5 Results:")
    print("Custom Minkowski (p=3):", d_custom)
    print("Scipy Minkowski  (p=3):", d_scipy)

    #A6
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\nA6 Results:")
    print("Train size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])

    #A7
    knn_model = train_knn_sklearn(X_train, y_train, k=3)
    print("\nA7 Result: kNN model trained with k=3")

    #A8
    acc_test = knn_accuracy_sklearn(knn_model, X_test, y_test)
    print("\nA& Result:")
    print("Test Accuracy (sklearn kNN):", acc_test)

    #A9

    y_pred_sklearn = knn_predictions_sklearn(knn_model, X_test)
    print("\nA9 Result:")
    print("Predictions (sklearn):", y_pred_sklearn[:10], "...")

    #A10
    y_pred_custom = knn_predict_all_custom(X_train, y_train, X_test, k=3)
    acc_custom = accuracy_custom(y_test, y_pred_custom)

    print("\nA10 Results:")
    print("custom kNN Accuracy:", acc_custom)

    #A11
    k_vals, acc_vals = accuracy_for_k_values_sklearn(X_train, y_train, X_test, y_test, 1, 11)

    print("\nA11 Results:")
    for k, a in zip(k_vals, acc_vals):
        print("k =", k, "accuracy=", a)

    plt.figure()
    plt.plot(k_vals, acc_vals, marker="o")
    plt.title("All Accuracy vs k (1 to 11)")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    #A12
    y_train_pred = knn_predictions_sklearn(knn_model, X_train)
    y_test_pred = knn_predictions_sklearn(knn_model, X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    train_metrics = precision_recall_f1_from_cm(cm_train)
    test_metrics = precision_recall_f1_from_cm(cm_test)

    print("\nA12 redults:")
    print("Confusion matrix (Train):\n", cm_train)
    print("Train Matrics (Acc, prec, Recall, F1):", train_metrics)
    print("Confusion Matrix (Test):\n", cm_test)
    print("Test Metrics (Acc, prec, Recall, F1):", test_metrics)

    #A13

    positive_label = 1
    cm_custom_test = confusion_matrix_custom(y_test, y_test_pred, positive_label)
    metrics_custom_test = metrics_from_cm_custom(cm_custom_test, beta=1.0)

    print("\nA13 Results:")
    print("Custom confusion Matrix (Test):\n", cm_custom_test)
    print("Custom Metrics (acc, prec, Recall, F1):", metrics_custom_test)

    #A14

    w = least_squares_classifier_train(X_train, y_train)
    y_pred_ls = least_squares_classifier_predict(X_test, w)
    acc_ls = accuracy_custom(y_test, y_pred_ls)

    print("\nA14 Results:")
    print("Least Squares Accuracy:", acc_ls)
    print("kNN (k=3) Accuracy    :",acc_test)

if __name__ == "__main__":
        main()

    
             

    
    
    


    
