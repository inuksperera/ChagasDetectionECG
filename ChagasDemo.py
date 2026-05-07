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
import streamlit as st
import wfdb
import os
import tempfile
import numpy as np
import numpy as np
from detect_disease import detect_disease
import torch
import os
import wfdb
from ecg_data import normalize_ecg_per_lead
from scipy.signal import resample

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
uploaded_files = st.file_uploader("Upload ECG files (.dat AND .hea)", type=['dat', 'hea'], accept_multiple_files=True)
if uploaded_files is not None:
    # st.info(f"File uploaded: {uploaded_files}")
    
    try:
        # Load the uploaded file as a numpy array
        # ecg_data = np.load(uploaded_file)
        # dummy_data = np.random.randn(1, 8, 2500).astype(np.float32)
        
        # # Display sample of the data (Optional viz if you want, but sticking to request)
        # if st.checkbox("Show ECG Signal Metadata"):
        #     st.write("Data Shape:", ecg_data.shape)

        signals = None
        fields = None
        ecg_input = None

        if uploaded_files:
            if len(uploaded_files) > 2:
                st.error("Please upload only 2 files (.dat and .hea).")
            elif len(uploaded_files) < 2:
                st.warning("Upload both .dat and .hea files.")
            elif len(uploaded_files) == 2:
                st.info("2 files uploaded correctly!")
                if uploaded_files:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Save all uploaded files to the temp directory
                        for up_file in uploaded_files:
                            with open(os.path.join(tmpdir, up_file.name), "wb") as f:
                                f.write(up_file.getbuffer())
                        
                        # Identify the base record name (the filename without extension)
                        # We look for the .dat file specifically to get the name
                        dat_files = [f for f in uploaded_files if f.name.endswith('.dat')]
                        
                        if dat_files:
                            record_name = os.path.join(tmpdir, os.path.splitext(dat_files[0].name)[0])
                            
                            try:
                                # --- ASSIGNMENT ---
                                signals, fields = wfdb.rdsamp(record_name)
                                
                            except Exception as e:
                                st.error(f"WFDB Error: {e}")
                        else:
                            st.warning("Please ensure you upload at least the .dat file.")

                # dummy_data = np.random.randn(1, 8, 2500).astype(np.float32)
                ecg_input = np.expand_dims(signals, axis=0).astype(np.float32)
                st.success(f"Successfully loaded {dat_files[0].name}")
                ecg_input = ecg_input.transpose(0, 2, 1)

                # Select only required leads (only uses 8 leads out of 12)
                ecg_input = np.concatenate((ecg_input[:, :2, :], ecg_input[:, 6:, :]), axis=1)
                    
                # Downsample if needed
                ecg_input = resample(ecg_input, 2500, axis=2)

                print("SHAPE AFTER REDUCED LEAD: " + str(ecg_input.shape))


                # apply normalization (Z-score) as expected by the trained model
                ecg_input = normalize_ecg_per_lead(ecg_input)


        if st.button("Run Prediction"):
            with st.spinner("Analyzing ECG with Deep Learning model..."):
                result = None
                result1 = None
                result2 = None
                # Use the path to trained model
                if (selected == 'MOL Comparison'):

                    combined_ckpt_path1 = './FINETUNED_WEIGHTS/checkpoint_linear_eval_combined_data_20260415-200225.pth'
                    result1 = detect_disease(
                    ecg_input=ecg_input,
                    combined_ckpt_path=combined_ckpt_path1, 
                    num_classes=1,  # 1 for binary classification (Chagas Positive/Negative)
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    threshold=0.5   # Probability threshold for positive prediction
                    )

                    combined_ckpt_path2 = './FINETUNED_WEIGHTS/checkpoint_linear_eval_combined_data_20260415-192106.pth'
                    result2 = detect_disease(
                    ecg_input=ecg_input,
                    combined_ckpt_path=combined_ckpt_path2, 
                    num_classes=1,  # 1 for binary classification (Chagas Positive/Negative)
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    threshold=0.5   # Probability threshold for positive prediction
                    )
                elif (selected == 'MOL Enabled'):
                    combined_ckpt_path = './FINETUNED_WEIGHTS/mol.pth'
                    result = detect_disease(
                    ecg_input=ecg_input,
                    combined_ckpt_path=combined_ckpt_path, 
                    num_classes=1,  # 1 for binary classification (Chagas Positive/Negative)
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    threshold=0.5   # Probability threshold for positive prediction
                    )
                elif (selected == 'MOL Disabled'):
                    combined_ckpt_path = './FINETUNED_WEIGHTS/ejepa.pth'
                    result = detect_disease(
                    ecg_input=ecg_input,
                    combined_ckpt_path=combined_ckpt_path, 
                    num_classes=1,  # 1 for binary classification (Chagas Positive/Negative)
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    threshold=0.5   # Probability threshold for positive prediction
                    )
                


                # Display Analysis Results
                st.markdown('<hr style="border: 1px solid white; opacity: 0.4;">', unsafe_allow_html=True)
                st.subheader("Diagnostic Results")

                #show comparison if MOL Comparison is selected
                if (selected == 'MOL Comparison'):

                    # Extract values for display
                    prob1 = float(result1['probability'].flatten()[0])
                    pred1 = int(result1['prediction'].flatten()[0])
                    prob2 = float(result2['probability'].flatten()[0])
                    pred2 = int(result2['prediction'].flatten()[0])

                    # Create two columns
                    left_col, right_col = st.columns(2)

                    # Left column - MOL Enabled
                    with left_col:
                        st.markdown(f"""
                        <div class="pred-box">

                        <h5 style="text-align:center;">MOL Enabled</h5>

                        - Chagas Positive Prediction: {prob1:.1%}
                        - Chagas Negative Prediction: {1.0 - prob1:.1%}
                        
                        </div>
                        """, unsafe_allow_html=True)

                        if pred1 == 1:
                            st.success("Result: Chagas Positive")
                        else:
                            st.error("Result: Chagas Negative")

                    # Right column - MOL Disabled
                    with right_col:
                        st.markdown(f"""
                        <div class="pred-box">

                        <h5 style="text-align:center;">MOL Disabled</h5>

                        - Chagas Positive Prediction: {prob2:.1%}   
                        - Chagas Negative Prediction: {1.0 - prob2:.1%}

                        </div>
                        """, unsafe_allow_html=True)
                            
                        if pred2 == 1:
                            st.success("Result: Chagas Positive")
                        else:
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
                    # Extract values for display
                    prob = float(result['probability'].flatten()[0])
                    pred = int(result['prediction'].flatten()[0])

                    # Display Analysis Results
                    st.markdown('<hr style="border: 1px solid white; opacity: 0.4;">', unsafe_allow_html=True)
                    st.subheader("Diagnostic Results")

                    # Create two columns
                    left_col2, right_col2 = st.columns(2)

                    # Left column - metric for positive prediction
                    with left_col2:
                        st.metric(label="Chagas Positive Prediction", value=f"{prob:.1%}")

                    # Right column - metric for negative prediction
                    with right_col2:
                        st.metric(label="Chagas Negative Prediction", value=f"{(1.0 - prob):.1%}")
                    
                    if pred == 1:
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

