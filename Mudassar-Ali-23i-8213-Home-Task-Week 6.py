import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardize the data (necessary for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Helper function to plot loss curves
def plot_loss_curve(loss_curve, title):
    plt.plot(loss_curve)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

# 4. Train the model using SGD optimizer
sgd_clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, solver='sgd', random_state=42)
sgd_clf.fit(X_train, y_train)

# Plot loss curve for SGD
plot_loss_curve(sgd_clf.loss_curve_, 'SGD Loss Curve')

# 5. Train the model using Adam optimizer
adam_clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, solver='adam', random_state=42)
adam_clf.fit(X_train, y_train)

# Plot loss curve for Adam
plot_loss_curve(adam_clf.loss_curve_, 'Adam Loss Curve')

# 6. Predict and compute confusion matrix for both models
y_pred_sgd = sgd_clf.predict(X_test)
y_pred_adam = adam_clf.predict(X_test)

conf_matrix_sgd = confusion_matrix(y_test, y_pred_sgd)
conf_matrix_adam = confusion_matrix(y_test, y_pred_adam)

# Plot confusion matrices
ConfusionMatrixDisplay(conf_matrix_sgd, display_labels=digits.target_names).plot()
plt.title('Confusion Matrix - SGD')
plt.show()

ConfusionMatrixDisplay(conf_matrix_adam, display_labels=digits.target_names).plot()
plt.title('Confusion Matrix - Adam')
plt.show()

# 7. Compare the performance by printing scores
print(f"SGD Score: {sgd_clf.score(X_test, y_test)}")
print(f"Adam Score: {adam_clf.score(X_test, y_test)}")
