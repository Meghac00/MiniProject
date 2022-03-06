import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
st.write("""
          # Pneumonia Prediction
          """
          )
upload_file = st.sidebar.file_uploader("Upload X-ray Images")
Generate_pred=st.sidebar.button("Predict")
model=tf.keras.models.load_model('model_alpha1.h5')
def import_n_pred(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size)
    image = ImageOps.grayscale(image)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('X_ray Image', expanded = True):
        st.image(image, width=350)#, use_column_width=True
    pred=import_n_pred(image, model)
    labels = ['NORMAL','PNEUMONIA']
    st.title("Prediction:{0}".format(labels[int(round(pred[0][0]))]))
