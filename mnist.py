import tensorflow as tf
from os import path, getcwd, chdir

print(tf.__version__)

# Get the mnist here
# !wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
path = f"{getcwd()}/mnist.npz"


def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accurancy') > 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images,
                                         test_labels) = mnist.load_data(path=path)
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images,
                        training_labels,
                        epochs=2,
                        callbacks=[callbacks]
                        )
    # Save Modell
    model.save('model.h5')

    return history.epoch, history.history['acc'][-1]


train_mnist_conv()
