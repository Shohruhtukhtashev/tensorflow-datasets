import streamlit as st
import tempfile
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import cv2

st.title("Tensorflow models")
st.write("Select the desired model")

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    models = st.radio(
    "Select the desired model",
    ('DeepWeeds', 'MNIST', 'Cats&Dogs'))

#------------          DeepWeeds       -------------------

if models == 'DeepWeeds':
    st.title("DeepWeeds")
    st.write(f'The DeepWeeds dataset consists of 17,509 images capturing eight different weed species native to Australia in situ with neighbouring flora.The \
        selected weed species are local to pastoral grasslands across the state of Queensland.The images were collected from weed infestations at the following \
        sites across Queensland: "Black River", "Charters Towers", "Cluden", "Douglas", "Hervey Range", "Kelso", "McKinlay" and "Paluma".')
    st.write(f'Can identify these things ➡ "Chinee apple", "Snake weed", "Lantana", "Prickly acacia", "Siam weed", "Parthenium", "Rubber vine", "Parkinsonia", "Nigatives"')

    model = tf.keras.models.load_model('../deepweeds/deepweeds.h5')

    if uploaded_file is not None:
        # for 3 channels image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(opencv_image,(256,256),interpolation=cv2.INTER_AREA)
        pre = np.argmax(model.predict(np.expand_dims(img,0)))
        label = {
                    0:'Chinee apple',
                    1:'Lantana',
                    2:'Parkinsonia',
                    3:'Parthenium',
                    4:'Prickly acacia',
                    5:'Rubber vine',
                    6:'Siam weed',
                    7:'Snake weed',
                    8:'Nigatives',
        }
        cap = label[pre]
        st.title(f"This is {cap}")
        st.image(opencv_image, channels='BGR', caption=f"Model predict: {cap}")


#------------     MNIST            -------------------

if models == 'MNIST':
    st.title("MNIST")
    st.write(f'The MNIST database of handwritten digits.')
    st.write(f'Can identify these things ➡ 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |')

    model = tf.keras.models.load_model('../mnist/mnist.h5')

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        image_g = img.convert("L")
        image = np.asarray(image_g).astype('uint8')
        
        #opencv_image = cv2.imdecode(file_bytes,1)
        image = cv2.resize(image,(28,28),interpolation=cv2.INTER_AREA)
        pre = np.argmax(model.predict(np.expand_dims(image,0)))
        st.title(f'This is {pre}')
        st.image(img, caption=f"Model predict: {pre}")


#------------          DeepWeeds       -------------------

if models == 'Cats&Dogs':
    st.title("Cats&Dogs")
    st.write(f'A large set of images of cats and dogs. There are 1738 corrupted images that are dropped.')
    st.write(f'Can identify these things ➡ Cats and Dogs')

    model = tf.keras.models.load_model('../cats_and_dogs/cats&dogs.h5')

    if uploaded_file is not None:
        # for 3 channels image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(opencv_image,(32,32),interpolation=cv2.INTER_AREA)
        p = model.predict(np.expand_dims(img,0))
        pre = np.round(model.predict(np.expand_dims(img,0))).item()
        label = {
                    0:'Cat',
                    1:'Dog',
        }
        st.write(f"{p}")
        cap = label[pre]
        st.title(f"This is {cap}")
        st.image(opencv_image, channels='BGR', caption=f"Model predict: {cap}")

