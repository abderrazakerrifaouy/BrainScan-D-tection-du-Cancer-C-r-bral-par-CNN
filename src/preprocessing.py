import numpy as np
import cv2 as cv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, image_size=(224, 224), path=None, classetr=None):

        self.image_size = image_size

        self.path = Path(path) if path is not None else None

        self.classetr = classetr if classetr is not None else []

        self.images = np.array([])
        self.labels = np.array([])

        self.valid_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }

        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None

    def load_images(self):
        images = []
        labels = []

        if self.path is None:
            raise ValueError("Le chemin du dataset n'est pas défini.")

        if not self.path.exists():
            raise FileNotFoundError(
                f"Le dossier du dataset n'existe pas : {self.path}"
            )

        for class_name in self.classetr:
            class_path = self.path / class_name

            if not class_path.is_dir():
                print(f"Warning: Dossier introuvable : {class_path}")
                continue

            for image_file in class_path.iterdir():

                if image_file.suffix.lower() not in self.valid_extensions:
                    print(f"Extension invalide : {image_file}")
                    continue

                if not image_file.is_file():
                    continue

                image = cv.imread(str(image_file))

 
                if image is None or image.size == 0:
                    print(f"Image illisible ou corrompue : {image_file}")
                    continue


                image = cv.resize(image, self.image_size)

                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


                images.append(image)
                labels.append(class_name)


        if images:
            self.images = np.array(images, dtype=np.uint8)
        else:
            self.images = np.empty(
                (0, self.image_size[1], self.image_size[0], 3),
                dtype=np.uint8
            )

        self.labels = np.array(labels)

        return self.images, self.labels

    def encode_labels(self):
        if self.labels.size == 0:
            raise ValueError("Aucun label à encoder. Chargez d'abord les images.")

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.labels)
        self.labels = encoded_labels

    def normalize_images(self):
        if self.images.size == 0:
            raise ValueError("Aucune image à normaliser. Chargez d'abord les images.")

        self.images = self.images.astype(np.float32) / 255.0

    def split_data(self, test_size=0.2, random_state=42):
        if self.images.size == 0 or self.labels.size == 0:
            raise ValueError("Aucune donnée à diviser. Chargez d'abord les images et les labels.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.images,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        return X_train, X_test, y_train, y_test

    def get_train_data(self):
        if self._X_train is None or self._y_train is None:
            raise ValueError("Les données d'entraînement n'ont pas été générées. Appelez d'abord split_data().")
        return self._X_train, self._y_train
    def get_test_data(self):
        if self._X_test is None or self._y_test is None:
            raise ValueError("Les données de test n'ont pas été générées. Appelez d'abord split_data().")
        return self._X_test, self._y_test

    
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

    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)

    print("y_train :", y_train.shape)
    print("y_test  :", y_test.shape)