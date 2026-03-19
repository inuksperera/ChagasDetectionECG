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
import numpy as np
import torch
from detect_disease import detect_disease

# st.title("Chagas Disease Detection from reduced 8-Lead ECG Using Deep Learning")
# st.markdown("""
# # Chagas-JEPA

# ## Chagas Disease Detection from reduced 8-Lead ECG Using Deep Learning
# """)
st.markdown("""
<h1 style='text-align: center; margin-top: -40px;'>Chagas-JEPA</h1>

<h3 style='text-align: center;'>
Chagas Disease Detection from reduced <br/> 8-Lead ECG Using Deep Learning
</h3>

<hr style="border: 1px solid white; opacity: 0.4;">
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    selected = option_menu('Chagas-JEPA',
                          #Chagas Disease Prediction System
                          ['MOL Comparison', 'MOL Enabled', 'MOL Disabled'],
                        icons=['grid', 'stack', 'layers'],
                        default_index=0)

    st.markdown("""
<style>
.sidebar-footer {
    position: fixed;
    bottom: 10px;
    left: 0;
    width: 21rem;
    display: flex;
    justify-content: center;
}

.sidebar-footer a {
    text-decoration: none;
    color: white;
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>

<div class="sidebar-footer">
<footer>
    <div style='display:flex; justify-content:center;'>
        <p style="font-size:1.1rem; margin:0;margin-bottom:1rem;">
            <a href="https://github.com/inuksperera/ChagasDetectionECG" target="_blank">
                Made by Inuka Perera &nbsp;
                <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                    -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                    .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12
                    0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
                    .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82
                    .44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15
                    0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                    0 1.07-.01 1.93-.01 2.2
                    0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8
                    c0-4.42-3.58-8-8-8z"/>
                </svg>
            </a>
        </p>
    </div>
</footer>
</div>
""", unsafe_allow_html=True)
    
if (selected == 'MOL Comparison'):
    st.write('#### Mixture-Of-Layers Aggregation Comparison')
elif (selected == 'MOL enabled'):
    st.write('#### Mixture-Of-Layers Aggregation Enabled')
elif (selected == 'MOL disabled'):
    st.write('#### Mixture-Of-Layers Aggregation Disabled')

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

    [data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label > div,
[data-testid="stSidebar"] [role="radiogroup"] label > div > div,
[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"],
[data-testid="stSidebar"] [role="radiogroup"] div[data-testid*="stMarkdownContainer"],
[data-testid="stSidebar"] [role="radiogroup"] .st-emotion-cache-* {{
    border-radius: 12px !important;          /* match your desired roundness – try 10px / 0.75rem / 1rem */
    overflow: hidden !important;             /* THIS IS THE KEY: clips any inner square content */
    background-clip: padding-box !important; /* helps bg respect the radius */
}}


    

    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

# set_background('bg_image_1.jpg')
set_background('bg_image_2.png')


# Include the CSS styling
st.markdown("""
<style>
.pred-box {
    background: rgba(15, 35, 80, 0.65);
    border: 1px dashed #00f0ff;
    border-radius: 10px;
    padding: 1rem;
    color: white;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# File uploading
uploaded_file = st.file_uploader("Upload an ECG sample ()")
if uploaded_file is not None:
    st.info(f"File uploaded: {uploaded_file.name}")
    
    try:
        # Load the uploaded file as a numpy array
        # ecg_data = np.load(uploaded_file)
        dummy_data = np.random.randn(1, 8, 2500).astype(np.float32)
        
        # # Display sample of the data (Optional viz if you want, but sticking to request)
        # if st.checkbox("Show ECG Signal Metadata"):
        #     st.write("Data Shape:", ecg_data.shape)

        if st.button("Run Prediction"):
            with st.spinner("Analyzing ECG with Deep Learning model..."):
                # Use the path to your trained linear eval model
                combined_ckpt = './downstream_tasks/output/linear_eval/checkpoint_linear_eval_final.pth'
                
                result = detect_disease(
                    ecg_input=dummy_data,
                    combined_ckpt_path=combined_ckpt,
                    num_classes=5,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )


                # Display Analysis Results
                st.markdown('<hr style="border: 1px solid white; opacity: 0.4;">', unsafe_allow_html=True)
                st.subheader("Diagnostic Results")

                #show comparison if MOL Comparison is selected
                if (selected == 'MOL Comparison'):

                    # Create two columns
                    left_col, right_col = st.columns(2)

                    # Left column - all content inside one div
                    with left_col:
                        st.markdown(f"""
                        <div class="pred-box">

                        <h5 style="text-align:center;">MOL Enabled</h5>

                        - Chagas Positive Prediction: 60.0%
                        - Chagas Negative Prediction: 40.0%

                        </div>
                        """, unsafe_allow_html=True)

                        st.success("Result: Chagas Positive")

                    # Right column - all content inside one div
                    with right_col:
                        st.markdown(f"""
                        <div class="pred-box">

                        <h5 style="text-align:center;">MOL Disabled</h5>

                        - Chagas Positive Prediction: 45.0%   
                        - Chagas Negative Prediction: 55.0%

                        </div>
                        """, unsafe_allow_html=True)
                            
                        st.error("Result: Chagas Negative")
                    
   

                    

                    
                    # if 'predicted_disease' in result:
                    #     # Index 0 if batch size is 1
                    #     main_diagnosis = result['predicted_disease'][0] if isinstance(result['predicted_disease'], list) else result['predicted_disease']
                        
                    #     st.success(f"Primary Detection: **{main_diagnosis}**")
                        
                    #     st.write("### Prediction Breakdown")
                        
                    #     top_diseases = result['top_diseases'][0] if isinstance(result['top_diseases'][0], list) else result['top_diseases']
                    #     top_probs = result['top_probs'][0] if isinstance(result['top_probs'], np.ndarray) and result['top_probs'].ndim > 1 else result['top_probs']
                        
                    #     cols = st.columns(len(top_diseases))
                    #     for idx, (disease, prob) in enumerate(zip(top_diseases, top_probs)):
                    #         with cols[idx]:
                    #             st.metric(label=disease, value=f"{prob:.1%}")
                                
                    #     st.info("Mapping: NORM (Normal), MI (Infarction), STTC (ST/T Change), CD (Conduction), HYP (Hypertrophy)")
                    # else:
                    #     st.write("Prediction indices:", result['prediction'])
                    #     st.write("Probability Distribution:", result['probability'])
                        


                #otherwise only show selected MOL setting results
                else:
                
                    if 'predicted_disease' in result:
                        # Index 0 if batch size is 1
                        main_diagnosis = result['predicted_disease'][0] if isinstance(result['predicted_disease'], list) else result['predicted_disease']
                        
                        st.success(f"Primary Detection: **{main_diagnosis}**")
                        
                        st.write("### Prediction Breakdown")
                        
                        top_diseases = result['top_diseases'][0] if isinstance(result['top_diseases'][0], list) else result['top_diseases']
                        top_probs = result['top_probs'][0] if isinstance(result['top_probs'], np.ndarray) and result['top_probs'].ndim > 1 else result['top_probs']
                        
                        cols = st.columns(len(top_diseases))
                        for idx, (disease, prob) in enumerate(zip(top_diseases, top_probs)):
                            with cols[idx]:
                                st.metric(label=disease, value=f"{prob:.1%}")
                                
                        st.info("Mapping: NORM (Normal), MI (Infarction), STTC (ST/T Change), CD (Conduction), HYP (Hypertrophy)")
                    else:
                        st.write("Prediction indices:", result['prediction'])
                        st.write("Probability Distribution:", result['probability'])





                                       # Display Analysis Results
                    st.markdown('<hr style="border: 1px solid white; opacity: 0.4;">', unsafe_allow_html=True)
                    st.subheader("Diagnostic Results")


                    # Create two columns
                    left_col2, right_col2 = st.columns(2)

                    # Left column - all content inside one div
                    with left_col2:
                        st.metric(label="Chagas Positive Prediction", value=f"{prob:.1%}")

                    with right_col2:
                        st.metric(label="Chagas Negative Prediction", value=f"{(1-prob):.1%}")
                    
                    if prob > (1-prob):
                        st.success("Result: Chagas Positive")
                    else:
                        st.error("Result: Chagas Negative")

                    # cols = st.columns(len(top_diseases))
                    #     for idx, (disease, prob) in enumerate(zip(top_diseases, top_probs)):
                    #         with cols[idx]:
                    #             st.metric(label=disease, value=f"{prob:.1%}")
                                
                    #     st.info("Mapping: NORM (Normal), MI (Infarction), STTC (ST/T Change), CD (Conduction), HYP (Hypertrophy)")
                    
                    
    except Exception as e:
        st.error(f"Error processing file or running inference: {e}")

