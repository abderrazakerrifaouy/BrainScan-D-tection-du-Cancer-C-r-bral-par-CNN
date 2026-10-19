from tensorflow import keras
from tensorflow.keras import layers
from preprocessing import Preprocessor

class Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=self.input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)



if __name__ == "__main__":

    preprocessor = Preprocessor(
            image_size=(224, 224),
            path="./Data",
            classetr=[
                "glioma",
                "meningioma",
                "pituitary",
                "notumor"
            ]
        )
    
    images, labels = preprocessor.load_images()

    preprocessor.encode_labels()
    preprocessor.normalize_images()
    preprocessor.split_data(test_size=0.2, random_state=42)
    X_train, y_train = preprocessor.get_train_data()
    X_test, y_test = preprocessor.get_test_data()

    model = Model(input_shape=(224, 224, 3), num_classes=len(set(labels)))
    model.compile_model()
    model.train(X_train, y_train, epochs=10)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
