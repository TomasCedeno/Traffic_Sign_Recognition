from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Configuracion camara
frameWidth = 480
frameHeight = 480
brightness = 180
font = cv2.FONT_HERSHEY_SIMPLEX

webcam = cv2.VideoCapture(0)
webcam.set(3, frameWidth)
webcam.set(4, frameHeight)
webcam.set(10, brightness)

#Cargar el modelo entrenado
model = load_model('model_trained.h5')

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img =cv2.equalizeHist(img)
    img = img/255
    return img

#Señales de transito (el indice corresponde al número de la predicción)
classes = [
    'Limite de Velocidad 20 km/h',
    'Limite de Velocidad 30 km/h',
    'Limite de Velocidad 50 km/h',
    'Limite de Velocidad 60 km/h',
    'Limite de Velocidad 70 km/h',
    'Limite de Velocidad 80 km/h',
    'Limite de Velocidad 100 km/h',
    'Limite de Velocidad 120 km/h',
    'Adelantar Carros',
    'Adelantar Camiones',
    'Pare',
    'Vehiculos Pesados',
    'No Pase',
    'Deslizamientos',
    'Ninos en la Via',
    'Animales en la Via'
]

while webcam.isOpened():

    #leer frame de webcam 
    success, imgOriginal = webcam.read()

    #Procesar imagen
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    cv2.putText(imgOriginal, "Clase:", (170, 35), font, 0.75, (0,255,0), 2, cv2.LINE_AA)

    #Hacer Prediccion    
    predictions = model.predict(img)[0]
    signClass = np.argmax(predictions)
    probability = np.max(predictions)

    if probability > 0.8:
        cv2.putText(imgOriginal, f"{classes[signClass]}", (270, 35), font, 0.75, (0,255,0), 2, cv2.LINE_AA)


    # Mostrar Salida
    cv2.imshow("Deteccion de Senales de Transito", imgOriginal)

    # Presionar 'Q' para terminar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar recursos
webcam.release()
cv2.destroyAllWindows()