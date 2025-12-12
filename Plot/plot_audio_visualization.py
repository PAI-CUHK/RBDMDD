import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_wav(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except FileNotFoundError:
        return

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title('Waveform') 
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 1, 2)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram (Log Frequency)') 
    
    plt.tight_layout()
    
    output_filename = "audio_visualization.png"
    plt.savefig(output_filename, dpi=300) 
    # ---------------------------------------------------------
