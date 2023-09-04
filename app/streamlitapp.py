import streamlit as st
import os
import imageio

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')
st.title('LipNet')

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

col_1, col_2 = st.columns(2)

if options:
    with col_1:
        st.info('The video below displays the coverted video in mp4 format.')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
    
    with col_2:
        st.info('This is only part of video machine learning model sees.')
        print(file_path)
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif' ,width=400)
        
        st.info('This is the output of the machine learning model')
        model = load_model()
        y_hat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(y_hat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Convert the tokens into words')
        words = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode()
        st.text(words)

        
