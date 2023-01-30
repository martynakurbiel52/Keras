
# Implementacja obsługi ładowania i predykcji modelu

from audioop import reverse
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import cv2
import tensorflow
import numpy as np

from PyQt5.QtGui import QImage

def qimage_to_array(image: QImage):
    """
    Funkcja konwertująca obiekt QImage do numpy array
    """
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    numpy_array = np.array(ptr).reshape(image.height(), image.width(), 1)

    # wykorzystanie bibloteki OpenCV do wyświetlenia obrazu po konwersji
    cv2.imshow('Check if the function works!', numpy_array)
    return numpy_array
    

def predict(image: QImage, model):
    """
    Funkcja wykorzystująca załadowany model sieci neuronowej do predykcji znaku na obrazie 

    Należy dodać w niej odpowiedni kod do obsługi załadowanego modelu
    """
    model = get_model()

    numpy_array = qimage_to_array(image)

    # wykorzystanie bibloteki OpenCV do zmiany wielkości obrazu do wielkości obrazów używanych w zbiorze MNIST
    numpy_array = cv2.resize(numpy_array, (28,28)).reshape((1, 28*28))

    # wykorzystanie bibloteki OpenCV do wyświetlenia obrazu po konwersji
    cv2.imshow('Check if the function works!!', numpy_array)

    prediction = model.predict(numpy_array)

    prediction = prediction[0]

    result = max(enumerate(prediction), key=(lambda x: x[1]))

    return result[0]


def get_model():
    """
    Funkcja wczytująca nauczony model sieci neuronowej 
    
    Należy dodać w niej odpowiedni kod do wczytywania na modelu oraz wag
    """    
    model = tensorflow.keras.models.load_model('model.h5')
    return model 