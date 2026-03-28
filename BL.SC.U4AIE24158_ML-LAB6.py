# ============================================================
# 22AIE213 - Lab Session 06
# Answers for A1 to A7 in ONE SINGLE PYTHON CODE
# UPDATED FOR CSV FILE
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

# ============================================================
# DATASET LOADING
# ============================================================

# ✅ UPDATED PATH (YOUR FILE)
file_path = r"C:\Users\preet\Downloads\classData (1).csv"

# Fallback path
if not os.path.exists(file_path):
    file_path = "/mnt/data/classData (1).csv"

# ✅ CHANGED: read_csv instead of read_excel
df = pd.read_csv(file_path)

print("\n================ DATASET PREVIEW ================\n")
print(df.head())
print("\nColumns:", list(df.columns))

# ------------------------------------------------------------
# Target detection
# ------------------------------------------------------------
possible_target_cols = ["G", "C", "B", "A"]

if all(col in df.columns for col in possible_target_cols):
    df["Target"] = df[possible_target_cols].idxmax(axis=1)
    feature_cols = [c for c in df.columns if c not in possible_target_cols + ["Target"]]
    target_col = "Target"
else:
    target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

X_original = df[feature_cols].copy()
y = df[target_col].copy()

print("\nTarget column used:", target_col)
print("Feature columns used:", feature_cols)

# ============================================================
# A1. ENTROPY
# ============================================================

def equal_width_binning(series, bins=4):
    series = pd.Series(series).dropna()
    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        return pd.Series(["Bin_1"] * len(series), index=series.index)

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    labels = [f"Bin_{i+1}" for i in range(bins)]
    return pd.cut(series, bins=bin_edges, labels=labels, include_lowest=True, duplicates="drop")

def entropy(values):
    values = pd.Series(values)
    probs = values.value_counts(normalize=True)
    return -sum(p * math.log2(p) for p in probs if p > 0)

def bin_features_equal_width(df_features, bins=4):
    out = pd.DataFrame(index=df_features.index)
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            out[col] = equal_width_binning(df_features[col], bins=bins).astype(str)
        else:
            out[col] = df_features[col].astype(str)
    return out

X_binned_equal_width = bin_features_equal_width(X_original, bins=4)
dataset_entropy = entropy(y)

print("\n================ A1. ENTROPY ================\n")
print("Entropy of target =", dataset_entropy)

# ============================================================
# A2. GINI INDEX
# ============================================================

def gini_index(values):
    values = pd.Series(values)
    probs = values.value_counts(normalize=True)
    return 1 - sum(p ** 2 for p in probs)

dataset_gini = gini_index(y)

print("\n================ A2. GINI INDEX ================\n")
print("Gini index of target =", dataset_gini)

# ============================================================
# A3. ROOT NODE USING INFORMATION GAIN
# ============================================================

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0

    for val in data[feature].unique():
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy

def find_root_node(data, features, target):
    gains = {}
    for feature in features:
        gains[feature] = information_gain(data, feature, target)

    root_feature = max(gains, key=gains.get)
    return root_feature, gains

data_for_root = X_binned_equal_width.copy()
data_for_root[target_col] = y

root_feature, all_gains = find_root_node(data_for_root, feature_cols, target_col)

print("\n================ A3. ROOT NODE ================\n")
for f, g in all_gains.items():
    print(f"{f} --> Information Gain = {g}")
print("\nRoot node =", root_feature)

# ============================================================
# A4. GENERAL BINNING
# ============================================================

def binning(series, bins=4, method="equal_width"):
    s = pd.Series(series).dropna()

    if s.nunique() == 1:
        return pd.Series(["Bin_1"] * len(s), index=s.index)

    labels = [f"Bin_{i+1}" for i in range(bins)]

    if method == "equal_width":
        edges = np.linspace(s.min(), s.max(), bins + 1)
        return pd.cut(s, bins=edges, labels=labels, include_lowest=True, duplicates="drop")

    elif method == "equal_frequency":
        return pd.qcut(s, q=bins, labels=labels, duplicates="drop")

def bin_dataset(df_features, bins=4, method="equal_width"):
    out = pd.DataFrame(index=df_features.index)
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            out[col] = binning(df_features[col], bins=bins, method=method).astype(str)
        else:
            out[col] = df_features[col].astype(str)
    return out

X_binned = bin_dataset(X_original)

print("\n================ A4. BINNING DONE ================\n")
print(X_binned.head())

# ============================================================
# A5. CUSTOM DECISION TREE
# ============================================================

class MyDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        data = X.copy()
        data["Target"] = y.values
        self.tree = self._build_tree(data, X.columns.tolist(), "Target", 0)

    def _majority(self, y):
        return Counter(y).most_common(1)[0][0]

    def _build_tree(self, data, features, target, depth):
        if len(data[target].unique()) == 1:
            return data[target].iloc[0]

        if not features or (self.max_depth and depth >= self.max_depth):
            return self._majority(data[target])

        best_feature, gains = find_root_node(data, features, target)

        tree = {best_feature: {}}

        for val in data[best_feature].unique():
            subset = data[data[best_feature] == val]
            subtree = self._build_tree(
                subset,
                [f for f in features if f != best_feature],
                target,
                depth + 1
            )
            tree[best_feature][val] = subtree

        return tree

    def predict(self, X):
        def predict_one(row, tree):
            if not isinstance(tree, dict):
                return tree
            root = next(iter(tree))
            val = row[root]
            if val in tree[root]:
                return predict_one(row, tree[root][val])
            return list(tree[root].values())[0]

        return np.array([predict_one(row, self.tree) for _, row in X.iterrows()])

my_tree = MyDecisionTree(max_depth=4)
my_tree.fit(X_binned, y)

print("\n================ A5. CUSTOM TREE ================\n")
print(my_tree.tree)

# ============================================================
# A6. TREE VISUALIZATION
# ============================================================

y_encoded = pd.factorize(y)[0]

X_encoded = X_binned.copy()
for col in X_encoded.columns:
    X_encoded[col] = pd.factorize(X_encoded[col])[0]

clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf.fit(X_encoded, y_encoded)

plt.figure(figsize=(15, 8))
plot_tree(clf, feature_names=X_encoded.columns, filled=True)
plt.show()

# ============================================================
# A7. DECISION BOUNDARY
# ============================================================

top2 = sorted(all_gains.items(), key=lambda x: x[1], reverse=True)[:2]
top2_features = [f[0] for f in top2]

X2 = X_original[top2_features]
y2 = pd.factorize(y)[0]

clf2 = DecisionTreeClassifier(max_depth=4)
clf2.fit(X2, y2)

plt.figure()
DecisionBoundaryDisplay.from_estimator(clf2, X2, alpha=0.4)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y2)
plt.show()
