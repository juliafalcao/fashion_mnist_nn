import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

""" import dataset """
dataset = keras.datasets.fashion_mnist

(train_X, train_y), (test_X, test_y) = dataset.load_data()
# train_x and test_x: images
# train_y and test_y: labels (0-9)

labels = ["shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]

""" dataset analysis """
train_shape = train_X.shape
test_shape = test_X.shape
print(f"Train data shape: {train_shape[0]} images, {train_shape[1]}x{train_shape[2]} pixels each")
print(f"Test data shape: {test_shape[0]} images, {test_shape[1]}x{test_shape[2]} pixels each")

# colorplot
plt.figure()
plt.imshow(train_X[0])
plt.colorbar()
plt.grid(False)
plt.savefig("colorplot.png")
plt.close()

# display first 25 images and their labels below
plt.figure(figsize = (10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_X[i], cmap = plt.cm.binary)
    plt.xlabel(labels[train_y[i]])

plt.savefig("clothes.png")
plt.close()

""" preprocessing """
# as pixel values are in range 0-255, scale them to 0-1
train_X = train_X / 255.0
test_X = test_X / 255.0


""" model """
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax) # returns array of probability scores for each class
])

model.compile(optimizer = tf.train.AdamOptimizer(), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

""" train """
model.fit(train_X, train_y, epochs = 5)

""" test """
test_loss, test_acc = model.evaluate(test_X, test_y)
print(f"Test accuracy: {test_acc}")

""" predict """
predictions = model.predict(test_X)


def plot_image(i, predictions_matrix, true_labels, images):
    predictions_list, true_label, image = predictions_matrix[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap = plt.cm.binary)
    
    predicted_label = np.argmax(predictions_list)
    predicted_label_probability = np.max(predictions_list)

    if predicted_label == true_label: color = "blue"
    else: color = "red"
    
    plot_label = "{} -> {}% {}".format(labels[predicted_label], int(100 * predicted_label_probability), labels[true_label]) 
    plt.xlabel(plot_label, color = color)


def plot_prediction_bars(i, predictions_matrix, true_labels):
    predictions_list, true_label = predictions_matrix[i], true_labels[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plot = plt.bar(range(10), predictions_list, color = "grey")
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_list)
    plot[predicted_label].set_color("red")
    plot[true_label].set_color("blue")


def plot_some(rows = 5, columns = 3):
    plt.figure(figsize = (2*2*columns, 2*rows))
    
    for i in range(rows * columns):
        plt.subplot(rows, 2 * columns, 2 * i+1)
        plot_image(i, predictions, test_y, test_X)
        plt.subplot(rows, 2 * columns, 2 * i+2)
        plot_prediction_bars(i, predictions, test_y)

    plt.show()

plot_some()