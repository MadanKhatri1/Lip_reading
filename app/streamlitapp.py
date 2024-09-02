import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

st.set_page_config(layout='wide')

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Lip Reading")
    st.info("This application is originally developed from the LipNet deep learning model.")

st.title('LipNet Full Stack App')
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info("This video below is in format of mp4")
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info("This video below is in format of gif and this is what the model sees")
        
        video, annotation = load_data(tf.convert_to_tensor(file_path))

        # Ensure video is a list of ndimages
        images = [frame.numpy() for frame in video]

        imageio.mimsave('animation.gif', images, fps=10)

        st.image('animation.gif',width=400)

        st.info("This is the model's output")
        model = load_model()
        yhat= model.predict(tf.expand_dims(video, axis=0))
        decoder=tf.keras.backend.ctc_decode(yhat, [75],greedy=True)[0][0]
        st.text(decoder)
        
        #convert the output to text
        st.info("Decode raw tokens into words")
        converted_text=tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_text)
               