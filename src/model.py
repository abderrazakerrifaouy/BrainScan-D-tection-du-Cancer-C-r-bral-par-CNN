from pathlib import Path
import sys
from collections import Counter
import matplotlib.pyplot as plt

root = Path.cwd().parent
sys.path.insert(0, str(root / "src"))

from preprocessing import Preprocessor

preprocessor = Preprocessor(
        image_size=(224, 224),
        path= str(root / "Data" ),
        classetr=[
            "glioma",
            "meningioma",
            "pituitary",
            "notumor"
        ]
    )

images, labels = preprocessor.load_images()

from tensorflow import keras
from tensorflow.keras import layers



model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(set(labels)), activation='softmax')
])

preprocessor.encode_labels()
preprocessor.normalize_images()
preprocessor.split_data(test_size=0.2, random_state=42)
preprocessor.split_data_validation()
X_train, y_train = preprocessor.get_train_data()
X_test, y_test = preprocessor.get_test_data()
X_val, y_val = preprocessor.get_validation_data()
labels = preprocessor.labels

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32  , validation_data=(X_val, y_val)) 


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)