from keras.engine.saving import model_from_json
from image_processing import cut_and_warp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import logging
logging.getLogger('tensorflow').disabled = True
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\diego\AppData\Local\Tesseract-OCR\tesseract.exe"


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

img = cv2.imread("sudoku.jpg")
img = cut_and_warp(img, False)


img = cv2.resize(img, (img.shape[1], img.shape[1]), interpolation=cv2.INTER_AREA)
new_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# new_image = cv2.equalizeHist(new_image)
# cv2.imshow("algo", new_image)
# cv2.waitKey(0)


w = new_image.shape[1] // 9
h = new_image.shape[1] // 9
for x in range(9):
    for y in range(9):
        cropped = new_image[y * h + 4:y * h + h, x * w + 4:x * w + w]
        cropped = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        cropped = cv2.blur(cropped, (2, 2))

        plt.imshow(cropped, cmap='gray')

        cropped = cropped.reshape((1, 28, 28, 1))
        cropped = cropped.astype('float32') / 255

        predictions = model.predict(cropped)
        max_idx = np.argmax(predictions)
        print(max_idx, predictions)

        plt.show()



