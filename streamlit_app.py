import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="🎵 Audio Classification App",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .cat-prediction {
        background-color: #FFE5E5;
        border: 2px solid #FF6B6B;
    }
    .dog-prediction {
        background-color: #E5F3FF;
        border: 2px solid #4ECDC4;
    }
    .confidence-high { color: #27AE60; font-weight: bold; }
    .confidence-medium { color: #F39C12; font-weight: bold; }
    .confidence-low { color: #E74C3C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('best_audio_classifier.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'best_audio_classifier.pkl' and 'feature_scaler.pkl' are in the same directory.")
        return None, None

def extract_mfcc_features(audio_data, sr=22050, n_mfcc=40):
    """Extract MFCC features from audio data"""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def predict_audio_class(audio_data, model, scaler, sr=22050):
    """Predict the class of audio data"""
    try:
        # Extract MFCC features
        mfccs = extract_mfcc_features(audio_data, sr)
        
        # Scale features
        mfccs_scaled = scaler.transform(mfccs.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(mfccs_scaled)[0]
        probabilities = model.predict_proba(mfccs_scaled)[0]
        
        # Return results
        class_name = 'Cat' if prediction == 0 else 'Dog'
        confidence = max(probabilities)
        
        return {
            'class': class_name,
            'confidence': confidence,
            'probabilities': {'Cat': probabilities[0], 'Dog': probabilities[1]}
        }
    except Exception as e:
        return {'error': str(e)}

def plot_waveform(audio_data, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax)
    ax.set_title('Audio Waveform', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_data, sr):
    """Plot audio spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title('Spectrogram', fontsize=16, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_mfcc_features(mfccs):
    """Plot MFCC features"""
    fig = px.bar(
        x=list(range(1, len(mfccs) + 1)),
        y=mfccs,
        title='MFCC Features',
        labels={'x': 'MFCC Coefficient', 'y': 'Value'},
        color=mfccs,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        title_font_size=16,
        title_font_family="Arial Black",
        showlegend=False
    )
    return fig

def plot_prediction_probabilities(probabilities):
    """Plot prediction probabilities"""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.2%}' for v in values],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Arial Black')
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        title_font_size=16,
        title_font_family="Arial Black",
        xaxis_title='Class',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False,
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Cat vs Dog Audio Classifier 🐱🐶</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload an audio file to classify whether it contains a cat or dog sound!</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Controls")
    st.sidebar.markdown("### 📁 Upload Audio File")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a .wav, .mp3, .flac, or .m4a file"
    )
    
    # Display options
    st.sidebar.markdown("### 📊 Visualization Options")
    show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
    show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)
    show_mfcc = st.sidebar.checkbox("Show MFCC Features", value=True)
    
    # Model info
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.info("""
    **Model**: Random Forest Classifier
    **Accuracy**: 83.93%
    **Features**: 40 MFCC coefficients
    **Training Data**: 277 audio files
    """)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load audio file
            audio_data, sr = librosa.load(uploaded_file, sr=22050)
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<h2 class="sub-header">🎵 Audio Player</h2>', unsafe_allow_html=True)
                st.audio(uploaded_file, format='audio/wav')
                
                # Audio information
                duration = len(audio_data) / sr
                st.markdown(f"""
                **📋 Audio Information:**
                - **Duration**: {duration:.2f} seconds
                - **Sample Rate**: {sr} Hz
                - **Samples**: {len(audio_data)}
                """)
            
            with col2:
                st.markdown('<h2 class="sub-header">🔮 Prediction</h2>', unsafe_allow_html=True)
                
                # Make prediction
                with st.spinner('Analyzing audio...'):
                    result = predict_audio_class(audio_data, model, scaler, sr)
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display prediction
                    predicted_class = result['class']
                    confidence = result['confidence']
                    
                    # Confidence level styling
                    if confidence >= 0.8:
                        conf_class = "confidence-high"
                        conf_emoji = "🎯"
                    elif confidence >= 0.6:
                        conf_class = "confidence-medium"
                        conf_emoji = "⚡"
                    else:
                        conf_class = "confidence-low"
                        conf_emoji = "❓"
                    
                    # Prediction box
                    box_class = "cat-prediction" if predicted_class == "Cat" else "dog-prediction"
                    animal_emoji = "🐱" if predicted_class == "Cat" else "🐶"
                    
                    st.markdown(f"""
                    <div class="prediction-box {box_class}">
                        <h2>{animal_emoji} {predicted_class}!</h2>
                        <p class="{conf_class}">{conf_emoji} Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    st.plotly_chart(
                        plot_prediction_probabilities(result['probabilities']),
                        use_container_width=True
                    )
            
            # Visualizations
            st.markdown('<h2 class="sub-header">📈 Audio Analysis</h2>', unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tabs = []
            if show_waveform:
                tabs.append("🌊 Waveform")
            if show_spectrogram:
                tabs.append("🎨 Spectrogram")
            if show_mfcc:
                tabs.append("📊 MFCC Features")
            
            if tabs:
                tab_objects = st.tabs(tabs)
                
                tab_idx = 0
                if show_waveform:
                    with tab_objects[tab_idx]:
                        fig_wave = plot_waveform(audio_data, sr)
                        st.pyplot(fig_wave)
                        plt.close(fig_wave)
                    tab_idx += 1
                
                if show_spectrogram:
                    with tab_objects[tab_idx]:
                        fig_spec = plot_spectrogram(audio_data, sr)
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)
                    tab_idx += 1
                
                if show_mfcc:
                    with tab_objects[tab_idx]:
                        mfccs = extract_mfcc_features(audio_data, sr)
                        fig_mfcc = plot_mfcc_features(mfccs)
                        st.plotly_chart(fig_mfcc, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")
            st.info("Please try uploading a different audio file.")
    
    else:
        # Welcome message
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
            <h3>👋 Welcome to the Audio Classifier!</h3>
            <p style="font-size: 1.1rem; color: #666;">
                Upload an audio file using the sidebar to get started. <br>
                The AI will analyze the audio and predict whether it contains cat or dog sounds.
            </p>
            <div style="margin-top: 2rem;">
                <span style="font-size: 3rem;">🎵🐱🐶🤖</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample predictions showcase
        st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", "83.93%", "High")
        
        with col2:
            st.metric("Training Files", "277", "Balanced")
        
        with col3:
            st.metric("MFCC Features", "40", "Optimal")
        
        with col4:
            st.metric("Model Type", "Random Forest", "Robust")

if __name__ == "__main__":
    main()
