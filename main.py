import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

categories = ["Cat", "Dog"]

for category in categories:

    path = category
    label = categories.index(category)

    for img in os.listdir(path)[:1000]:

        img_path = os.path.join(path, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            data.append(image.flatten())
            labels.append(label)

        except:
            pass

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = SVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100)