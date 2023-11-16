import streamlit as st
import cv2
import numpy as np
#from PIL import Image, ImageOps #Install pillow instead of PIL
from PIL import Image as Imag, ImageOps as ImagOps
import numpy as np
from PIL import Image, ImageOps
import time
from keras.models import load_model
import paho.mqtt.client as paho
from IPython.display import Audio


broker="157.230.214.127"
port=1883
def on_publish(client,userdata,result):             #create function for callback
    #print("el dato ha sido publicado \n")
    pass


def on_message(client, userdata, message):
    global message_received
    time.sleep(2)
    message_received=str(message.payload.decode("utf-8"))
    print(message_received)


client1=paho.Client("Aplicacion2")
client1.on_publish = on_publish
client1.subscribe("Instructions")
client1.on_message = on_message
client1.connect(broker,port)

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    print(prediction)
    if prediction[0][0] >0.3:
       print('abrir: ')
       client1.publish("Instructions","{'Act1': 'abre la puerta'}",qos=0, retain=False)
       st.header('Abierto, con Probabilidad: '+str( prediction[0][0]) )
       #sound_file = 'hum_h.wav'
       #display(Audio(sound_file, autoplay=True))
       time.sleep(0.5)
    if prediction[0][1]>0.6:
       print('cerrar')
       client1.publish("Instructions","{'Act1': 'Cierra la Puerta'}",qos=0, retain=False)
       st.header('Cerrado, con Probabilidad: '+str( prediction[0][0]) )
       time.sleep(0.5)
    if prediction[0][2]>0.6:
       print('vacio')
       client1.publish("Instructions","{'Act1': 'Cierra la Puerta'}",qos=0, retain=False)
       st.header('Vacío, con Probabilidad: '+str( prediction[0][0]) )
       time.sleep(0.5)

