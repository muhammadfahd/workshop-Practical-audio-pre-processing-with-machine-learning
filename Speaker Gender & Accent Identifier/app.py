# ==========================================
# PRODUCTION CAPSTONE: AUDIO AI WEB PORTAL
# ==========================================
# Purpose: Streamlit application running scratch 2D-CNN for audio biometrics.

import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Set page configurations
st.set_page_config(page_title="Voice Analytics AI", page_icon="🎙️", layout="centered")

st.title("🎙️ AI Speaker Profiling & Biometrics")
st.write("Upload an audio file (.wav format) to analyze the acoustic signatures and predict the structural gender profile.")

# ==========================================
# 1. CUSTOM 2D-CNN ARCHITECTURE CORE
# ==========================================
class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super(AudioClassifierCNN, self).__init__()
        # Layer Block 1: Basic acoustic edges
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer Block 2: Complex structural pattern maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Linear fully connected mapping blocks
        self.fc1 = nn.Linear(32 * 16 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Cache the model loading execution sequence so it doesn't reload on every webpage refresh
@st.cache_resource
def load_trained_model():
    model = AudioClassifierCNN()
    # Loading compiled weights saved from Google Colab
    model.load_state_dict(torch.load("speaker_cnn_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_trained_model()
    # Configuration module for feature extraction transformation
    mel_transformer = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
except Exception as e:
    st.error("⚠️ Model file 'speaker_cnn_weights.pth' not found! Please ensure it is placed in the same directory as app.py")
    st.stop()

# ==========================================
# 2. USER INTERFACE (UI) LAYOUT DESIGN
# ==========================================
uploaded_file = st.file_uploader("Choose a standard WAV voice sample...", type=["wav"])

if uploaded_file is not None:
    st.success("🎉 Audio uploaded successfully!")
    
    # Audio Playback Widget
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("⚡ Processing audio frequencies..."):
        # DIRECT BYPASS: Using native soundfile engine to prevent Windows native runtime system crashes
        uploaded_file.seek(0)
        data, samplerate = sf.read(uploaded_file)
        
        # Convert numpy array signals to PyTorch FloatTensors [Channels, TimeSteps]
        if len(data.shape) == 1:
            live_waveform = torch.FloatTensor(data).unsqueeze(0)
        else:
            live_waveform = torch.FloatTensor(data).t()
            
        # Channel reduction if data stream is loaded in multi-track stereo format
        if live_waveform.shape[0] > 1:
            live_waveform = torch.mean(live_waveform, dim=0, keepdim=True)
            
        # Sample rate alignment layer to enforce standard model expectation (16000Hz)
        if samplerate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
            live_waveform = resampler(live_waveform)
            
        # Convert raw wave signals into dynamic 2D graphic spectrogram maps
        live_spec = mel_transformer(live_waveform)
        if live_spec.shape[2] >= 32:
            live_spec = live_spec[:, :, :32]
        else:
            live_spec = torch.nn.functional.pad(live_spec, (0, 32 - live_spec.shape[2]))
            
        # Forward Model Inference Validation Pass
        with torch.no_grad():
            output_logs = model(live_spec.unsqueeze(0))
            _, pred_idx = torch.max(output_logs, 1)
            
        # Target profile mapping dictionaries
        gender_map = {
            0: "Profile A (Male-dominant Acoustic Signatures)", 
            1: "Profile B (Female-dominant Acoustic Signatures)"
        }
        result = gender_map[pred_idx.item()]
        
        # Visual metrics rendering block
        st.markdown("---")
        st.subheader("🤖 AI Analysis Metrics")
        if pred_idx.item() == 0:
            st.info(f"Detected Speaker: **{result}**")
        else:
            st.success(f"Detected Speaker: **{result}**")
            
        # ==========================================
        # 3. INTERACTIVE HEATMAP PLOT RENDERING
        # ==========================================
        st.subheader("🖼️ Acoustic Graphic Visualizations")
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        live_spec_img = live_spec.squeeze().numpy()
        im = ax.imshow(live_spec_img, aspect='auto', origin='lower', cmap='magma')
        ax.set_title("Input 2D Mel-Spectrogram Map Processed by CNN", fontsize=10, fontweight='bold')
        ax.set_xlabel("Enforced Model Time Frames", fontsize=8)
        ax.set_ylabel("Mel Frequency Bins", fontsize=8)
        fig.colorbar(im, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        # Displaying matplotlib figures securely inside Streamlit layout frameworks
        st.pyplot(fig)