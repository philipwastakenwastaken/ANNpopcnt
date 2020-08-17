import numpy as np
from tensorflow import keras


np.random.seed(12938498)


def create_data(size):
    data = np.random.randint(0, (1 << 32) + 1, (size, 33))
    labels = []

    for i in range(size):
        x = data[i][0]
        x = bin(x)[2:]

        for j in range(1, 33):
            data[i][j] = 0

        for indx, c in enumerate(x):
            data[i][indx] = int(c)

        label = [0] * 32
        label[x.count("1") - 1] = 1
        labels.append(label)

    return data, np.array(labels).reshape(size, 32)


epoch_size = 10 ** 6

X_train, y_train = create_data(epoch_size)
X_test, y_test = create_data(int(epoch_size * 0.1))


inputs = keras.Input(X_train.shape[1], name="popcnt")
x = keras.layers.Dense(128, activation="relu")(inputs)
x = keras.layers.Dropout(.1)(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(y_train.shape[1], activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
     loss=keras.losses.CategoricalCrossentropy(),
     optimizer=keras.optimizers.Adam(),
     metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=16, epochs=10,
                    validation_split=0.2)

test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss", test_scores[0])
print("Test accuracy", test_scores[1])
