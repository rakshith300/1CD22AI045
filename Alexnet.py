from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam


class ImprovedAlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # -------- Block 1 --------
        self.add(Conv2D(64, (11,11), strides=4, padding='same',
                        input_shape=input_shape))
        self.add(BatchNormalization())
        self.add(DenseActivation := tf.keras.layers.Activation('relu'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=2))

        # -------- Block 2 --------
        self.add(Conv2D(192, (5,5), padding='same'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=2))

        # -------- Block 3 --------
        self.add(Conv2D(384, (3,3), padding='same'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))

        self.add(Conv2D(384, (3,3), padding='same'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))

        self.add(Conv2D(256, (3,3), padding='same'))
        self.add(BatchNormalization())
        self.add(tf.keras.layers.Activation('relu'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=2))

        # -------- Classifier --------
        self.add(Flatten())
        self.add(Dense(1024, activation='relu'))  # Reduced parameters
        self.add(Dropout(0.5))
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax'))


# Example usage
input_shape = (224, 224, 3)
num_classes = 1000

model = ImprovedAlexNet(input_shape, num_classes)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
