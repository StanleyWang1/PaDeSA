import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate the mean, variance and priors for each class
        self.mean = np.zeros((n_classes,X.shape[1]), dtype=np.float64)
        self.var = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, label in enumerate(self.classes):
            X_class = X[y == label]
            self.mean[idx, :] = X_class.mean(axis=0)
            self.var[idx, :] =X_class.var(axis=0)
            self.priors[idx] = X_class.shape[0] / float(X.shape[0])

    def predict(self, X):
        y_pred = [self._predict(sample) for sample in X]
        return np.array(y_pred)

    def _predict(self, sample):
        posteriors = []

        # Compute posterior probability for each class
        for idx, label in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self.pdf(idx, sample)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx] + 1e-10  # Adding a small constant to avoid division by zero
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def k_fold_cross_validation(X, y, k=5):
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    fold_size = len(X) // k
    accuracies = []

    for i in range(k):
        # Create train and validation splits
        validation_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, validation_indices)

        X_train, X_val = X[train_indices], X[validation_indices]
        y_train, y_val = y[train_indices], y[validation_indices]

        # Train the classifier
        clf = GaussianNaiveBayes()
        clf.fit(X_train, y_train)

        # Validate the classifier
        y_pred = clf.predict(X_val)
        accuracy =np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    return accuracies

# Load the data

#data = pd.read_csv("extracted_data.csv")
#X = data.drop("label", axis=1).values
#y = data["label"].map({"nut": 0, "bolt": 1, "washer": 2}).values

start_time = time.time()

data = pd.read_csv("training_x_600_23_features.csv")
X = data.drop(columns=["y", "x_10"]).values

y = data["y"].values


X, y = shuffle(X, y, random_state=0)

# normalizing the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Splitting data
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the classifier
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

accuracies = k_fold_cross_validation(X, y, k=5)
average_accuracy = np.mean(accuracies)

print(f"Accuracies for each fold: {accuracies}")
print(f"Average Accuracy: {average_accuracy:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Runtime: {elapsed_time:.2f} seconds")


def plot_feature_distribution(X, y, feature_index, class_labels, feature_names):
    colors = ['r', 'g', 'b']
    plt.figure(figsize=(10, 3))

    all_data = X[:, feature_index]
    mean = np.mean(all_data)
    std = np.std(all_data)

    # Calculate the bounds for certain percent of the data
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std

    for label, color in zip(class_labels, colors):
        data = X[y == label][:, feature_index]
        data = data[~np.isnan(data) & ~np.isinf(data) & ~np.isneginf(data)]

        # Only consider data within the bounds
        data = data[(data >= lower_bound) & (data <= upper_bound)]

        if data.size == 0:
            continue
        sns.histplot(data, color=color, label=class_labels[label], kde=True, bins=30)

    plt.title(f"{feature_names[feature_index]} Distribution", fontsize=14)
    plt.xlabel(feature_names[feature_index], fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f"Distribution_{feature_names[feature_index]}.png")
    #plt.show()


# Feature names
feature_names = ["Volume of 3D Part", "Surface Area of 3D Part",
                 "X Dimensions of Bounding Box", "Y Dimensions of Bounding Box", "Z Dimensions of Bounding Box",
                 "Volume of Bounding Box",
                 "X Coordinates of Bounding Box Center","Y Coordinates of Bounding Box Center", "Z Coordinates of Bounding Box Center",
                 "X Coordinates of Center of Mass", "Y Coordinates of Center of Mass", "Z Coordinates of Center of Mass",
                 "Number of Vertices in 3D Part", "Edges of 3D Part", "Faces of 3D part",
                 "x_y","Bounding Box Aspect Ratio Y vs. Z","z_x","packing","area ratio","nv_ne", "Number of Faces vs. Number of Vertices"]

# Class labels
class_labels = {0: "Bolts", 1: "Nuts", 2: "Washers"}

for i in range(X_train.shape[1]):
    plot_feature_distribution(X_train, y_train, i, class_labels, feature_names)


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.set(font_scale=4)
# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()