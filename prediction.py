import os
import streamlit as st
from PIL import Image
import torch
import pandas as pd

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
   # giving a title
   st.set_page_config(page_title='YOLO Classifier', page_icon='favicon.png')
   st.title('African Wildlife Animal Classifier')
   st.subheader('Upload either Buffalo/Elephant/Rhino/Zebra image for prediction')
   image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
   # code for Prediction
   prediction = ''
   st.session_state['answer'] = ''
   realans = ['', 'abc', 'edf']
   if  st.session_state['answer'] in realans:
           answerStat = "correct"
   elif st.session_state['answer'] not in realans:
           answerStat = "incorrect"
           
        
        

   # creating a button for Prediction

   if st.button('Predict'):
     if image_file is not None:
         # To See details
        with st.spinner('Loading Model and Image...'):
            model = torch.hub.load('yolov5','custom',path='best.pt',source='local')
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        st.write(file_details)
        img = load_image(image_file)
        st.image(img,width=640)
        # st.video(data, format="video/mp4", start_time=0)
        with st.spinner('Predicting...'):
            result=model(img,size=640)
            l= result.pandas().xyxy[0]['name']
        d={}
        for i in l:
          d[i]=d.get(i,0)+1
        s=""
        for i in d:
          s+=f"{d[i]} {i}, "
        st.success(s[:-2])
        st.image(Image.fromarray(result.render()[0]))



if __name__ == '__main__':
    main()

