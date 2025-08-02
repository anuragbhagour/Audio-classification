# 🎵 Audio Classification Streamlit Web App

A beautiful and interactive web interface for classifying cat and dog sounds using machine learning.

## 🌟 Features

- **📤 File Upload**: Support for multiple audio formats (WAV, MP3, FLAC, M4A)
- **🎵 Audio Player**: Built-in audio player to listen to uploaded files
- **🤖 AI Prediction**: Real-time classification with confidence scores
- **📊 Visualizations**: Interactive waveforms, spectrograms, and MFCC features
- **📈 Model Metrics**: Display of model performance and statistics
- **🎨 Beautiful UI**: Modern, responsive design with custom styling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Your trained model files (`best_audio_classifier.pkl` and `feature_scaler.pkl`)

### Installation

1. **Clone or download this repository**

2. **Install dependencies** (Choose one method):

   **Method 1: Using the setup script (Windows)**
   ```bash
   setup.bat
   ```

   **Method 2: Using pip directly**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**
   
   Make sure these files are in the project directory:
   - `best_audio_classifier.pkl`
   - `feature_scaler.pkl`
   
   *(These files are created when you run the Jupyter notebook)*

4. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## 🎮 How to Use

1. **Upload an audio file** using the sidebar file uploader
2. **Listen to the audio** using the built-in player
3. **View the AI prediction** with confidence scores
4. **Explore visualizations** including:
   - Audio waveform
   - Spectrogram (frequency analysis)
   - MFCC features (what the AI "sees")

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 83.93%
- **Features**: 40 MFCC (Mel-Frequency Cepstral Coefficients)
- **Training Data**: 277 audio files (cats and dogs)
- **Cross-validation**: 88.24% accuracy

## 🎨 Interface Features

### Main Components

1. **Header Section**: Welcome message and app title
2. **Sidebar Controls**: 
   - File uploader
   - Visualization toggles
   - Model information
3. **Prediction Area**: 
   - Audio player
   - Classification results
   - Confidence visualization
4. **Analysis Tabs**:
   - Waveform visualization
   - Spectrogram analysis
   - MFCC feature display

### Visual Indicators

- **🐱 Cat Prediction**: Red-themed with cat emoji
- **🐶 Dog Prediction**: Blue-themed with dog emoji
- **Confidence Levels**:
  - 🎯 High (≥80%): Green
  - ⚡ Medium (60-79%): Orange
  - ❓ Low (<60%): Red

## 📁 File Structure

```
Audio-classification/
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── setup.bat                 # Windows setup script
├── README.md                 # This file
├── best_audio_classifier.pkl # Trained model (generated)
├── feature_scaler.pkl        # Feature scaler (generated)
├── audio_classification.ipynb # Training notebook
└── data/                     # Training data directory
```

## 🛠️ Technical Details

### Dependencies

- **Streamlit**: Web app framework
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plots
- **NumPy/Pandas**: Data manipulation

### Audio Processing Pipeline

1. **Load Audio**: Using librosa with 22.05kHz sample rate
2. **Feature Extraction**: 40 MFCC coefficients
3. **Preprocessing**: StandardScaler normalization
4. **Prediction**: Random Forest classification
5. **Visualization**: Multiple analysis views

## 🎯 Supported Audio Formats

- **WAV** (recommended)
- **MP3**
- **FLAC**
- **M4A**

## 🚨 Troubleshooting

### Common Issues

1. **"Model files not found" error**
   - Ensure `best_audio_classifier.pkl` and `feature_scaler.pkl` are in the project directory
   - Run the Jupyter notebook to generate these files

2. **Audio file not loading**
   - Check if the file format is supported
   - Try converting to WAV format
   - Ensure file is not corrupted

3. **Package installation errors**
   - Use Python 3.8 or higher
   - Try installing packages individually
   - Use virtual environment if needed

### Performance Tips

- **Upload smaller audio files** (< 30 seconds) for faster processing
- **Use WAV format** for best compatibility
- **Enable only needed visualizations** to improve loading speed

## 🎉 Example Usage

1. Upload a cat meowing sound → Should predict "Cat" with high confidence
2. Upload a dog barking sound → Should predict "Dog" with high confidence
3. Upload mixed or unclear audio → Will show lower confidence scores

## 🔮 Future Enhancements

- Support for more animal sounds
- Real-time audio recording
- Batch file processing
- Audio trimming tools
- Enhanced visualizations
- Model comparison features

## 📞 Support

If you encounter any issues:

1. Check that all dependencies are installed correctly
2. Ensure model files are present and accessible
3. Verify audio file format compatibility
4. Try running with different audio files

---

**Happy Classifying! 🎵🐱🐶**
