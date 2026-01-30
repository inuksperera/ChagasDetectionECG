"""
streamlit tutorial: https://www.youtube.com/watch?v=8Q_QQVQ1HZA
streamlit tutorial github: https://github.com/shaadclt/Multiple-Disease-Prediction-System
alsdfjskdfj
for setting background image: https://stackoverflow.com/questions/76320197/streamlit-app-not-loading-background-image
background image: https://img.freepik.com/premium-vector/abstract-background-blue-futuristic-technology-world-maps-digital-ecg-heartbeat-pulse-line-wave-monitor_35887-478.jpg
"""

import streamlit as st
from streamlit_option_menu import option_menu
import base64

st.title("Chagas Detection Using 12-lead ECG")

# Sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
# Set background image using HTML and CSS
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(
            rgba(5, 15, 40, 0.88), 
            rgba(8, 25, 60, 0.92)
        ),
        url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .stApp {{
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0.0) !important;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(4, 15, 40, 0.7); 
        border-right: 1px solid rgba(0, 255, 255, 0.12);
        backdrop-filter: blur(4px);
    }}

    /* File uploader styling */
    [data-testid="stFileUploader"] {{
        background: rgba(15, 35, 80, 0.65);
        border: 1px dashed #00f0ff;
        border-radius: 10px;
        padding: 1.5rem;
    }}
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)


# set_background('bg_image_1.jpg')
set_background('bg_image_2.png')


# File uploading
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size (bytes):", uploaded_file.size)