import csv
import random
import numpy as np


np.random.seed(4)

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    return exps / (sum_exps + 1e-10)  # avoid division by zero

def predict(features, weights):
    logits = features.dot(weights)
    return softmax(logits)

def compute_loss(features, labels, weights, num_classes):
    N = len(features)
    logits = features.dot(weights)
    predictions = softmax(logits)
    label_onehot =np.eye(num_classes)[labels]
    # avoid log(0)
    loss = -np.sum(label_onehot * np.log(predictions + 1e-10)) / N
    return loss


def update_weights(features, labels, weights, lr, num_classes):
    N = len(features)
    logits = features.dot(weights)
    predictions = softmax(logits)
    label_onehot = np.eye(num_classes)[labels]
    gradient = -features.T.dot(label_onehot - predictions) / N
    weights -= lr * gradient
    return weights

def normalize_data(data):
    data = np.array(data)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / (stds + 1e-10)

# Loading and splitting data

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data


def split_data(data, split_ratio):
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]


def softmax_regression(filename, split_ratio=0.9, lr=0.00001, epochs=5000):
    headers, all_data = load_data(filename)

    # Map labels to integers
    unique_labels = list(set([row[-1] for row in all_data]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Convert data to float and labels to int
    data_list = normalize_data([np.array([1.0] + [float(value) for value in row[:-1]]) for row in all_data])  # Adding bias term and normalizing
    labels_list = [label_map[row[-1]] for row in all_data]

    # Split data
    train_data, test_data = split_data(data_list, split_ratio)
    train_labels, test_labels = split_data(labels_list, split_ratio)

    num_features = len(train_data[0])
    num_classes = len(unique_labels)

    # Initialize weights
    weights = np.zeros((num_features, num_classes))

    # Train
    for epoch in range(epochs):
        weights = update_weights(np.array(train_data), np.array(train_labels), weights, lr, num_classes)
        if epoch % 500 == 0:
            loss =compute_loss(np.array(train_data), np.array(train_labels), weights, num_classes)
            print(f"Epoch {epoch}, Loss: {loss}")

    # Test
    predictions = predict(np.array(test_data), weights)
    predicted_labels = np.argmax(predictions, axis=1)
    correct = np.sum(predicted_labels == test_labels)

    accuracy = correct / len(test_data)
    return accuracy, label_map


accuracy, label_map = softmax_regression("training_x_600.csv", split_ratio= 0.8)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Label mapping:", label_map)

