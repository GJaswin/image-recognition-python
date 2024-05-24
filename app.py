import streamlit as st
import json
import img_recognition
from img_recognition import predict, class_names

with open('flowers.json','r') as file:
    data = json.load(file)

flowers = list(data.keys())

st.header(":green[AI Flower Identifier]", divider = 'rainbow')
input_image=st.file_uploader("")

if input_image is not None:
    st.image(input_image, use_column_width = "auto")
    
    with st.spinner("Analysing..."):
        try:
            class_no, flower, probs = predict(input_image)
            st.subheader(":orange[{}]".format(flower), anchor = False)

            st.write(f"**Scientific Name:** *{data[flowers[class_no]]['sci_name']}*")
            st.write(f"**Family:**   *{data[flowers[class_no]]['family']}*")
            st.write(data[flowers[class_no]]['desc'])


            st.write("*:gray[Prediction Probabilities:]*")
            for i in range(0, len(class_names)):
                st.write(f"*:gray[{class_names[i]}: {probs[i]:.2f}%]*")
            
        except:
            st.subheader(":red[An Error Occurred!]")
            st.write("Try an image with a lower resolution")
            


else:
    st.subheader(":gray[Upload an image to get started]", anchor = False)
    st.write("### This AI Model supports detection of 5 types of flowers: ", anchor = False)
    # List of flowers
    '''
     - **Lilly**
     - **Lotus**
     - **Orchid**
     - **Sunflower**
     - **Tulip**
    ''' 

    st.write("Additionally, it will provide a description of the identified flower")

