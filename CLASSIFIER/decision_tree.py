import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def gini_impurity(labels):
    impurity = 1
    label_counts = Counter(labels)
    for label in label_counts:
        prob_of_label = label_counts[label] / len(labels)
        impurity -= prob_of_label ** 2
    return impurity


# split dataset
def split_dataset(data, labels, column, value):
    left_data, right_data, left_labels, right_labels = [], [], [], []
    for row, label in zip(data, labels):
        if row[column] < value:
            left_data.append(row)
            left_labels.append(label)
        else:
            right_data.append(row)
            right_labels.append(label)
    return left_data, right_data, left_labels, right_labels



class DecisionNode:
    def __init__(self, column, value, true_branch, false_branch):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class LeafNode:
    def __init__(self, labels):
        self.predictions = Counter(labels)


# build the tree
def build_tree(data, labels, depth=1, max_depth=5):
    if len(set(labels)) == 1:
        return LeafNode(labels)
    if depth == max_depth:
        return LeafNode(labels)

    best_gini = 1
    best_split = None
    current_gini = gini_impurity(labels)

    n_features = len(data[0])
    for col in range(n_features):
        values = set([row[col] for row in data])
        for value in values:
            left_data, right_data, left_labels, right_labels = split_dataset(data, labels, col, value)
            if len(left_data) == 0 or len(right_data) == 0:
                continue

            p_left = len(left_data) / len(data)
            gini = (p_left*gini_impurity(left_labels)) + ((1 - p_left) * gini_impurity(right_labels))

            if gini < best_gini:
                best_gini = gini
                best_split = (col, value, left_data, right_data, left_labels, right_labels)

    if best_gini == current_gini:
        return LeafNode(labels)

    true_branch = build_tree(best_split[2], best_split[4], depth + 1, max_depth)
    false_branch = build_tree(best_split[3], best_split[5], depth + 1, max_depth)

    return DecisionNode(best_split[0], best_split[1], true_branch, false_branch)

def split_data(data, split_ratio):
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

# predictions
def classify(row, node):
    if isinstance(node, LeafNode):
        return node.predictions.most_common(1)[0][0]

    if row[node.column] < node.value:
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# Load
dataset = pd.read_csv("training_x_600.csv")

# Convert to lists
X = dataset.drop(columns=['y']).values.tolist()
y = dataset['y'].tolist()

X, y = shuffle(X, y, random_state=0)

# Splitting the data into training and testing sets
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test the classifier and calculate accuracy
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
