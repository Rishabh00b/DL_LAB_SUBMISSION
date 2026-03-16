import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# ==============================
# 1 Load CIFAR10 Dataset
# ==============================

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Reduce dataset size (to avoid RAM error)
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# ==============================
# 2 Normalize Images
# ==============================

x_train = x_train / 255.0
x_test = x_test / 255.0

# ==============================
# 3 Class Names
# ==============================

class_names = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

# ==============================
# 4 Build CNN Model
# ==============================

model = models.Sequential([

    layers.Input(shape=(32,32,3)),

    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),

    layers.Dense(10,activation='softmax')

])

# ==============================
# 5 Compile Model
# ==============================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 6 Train Model
# ==============================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# ==============================
# 7 Evaluate Model
# ==============================

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)

# ==============================
# 8 Plot Accuracy Graph
# ==============================

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()