import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle  # If scaler is saved as a pickle file
import sys
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, lfilter
import soundfile as sf
from io import BytesIO

def calculate_pitch(y, sr):
    """Calculates the pitch of the audio signal."""
    return librosa.core.piptrack(y=y, sr=sr)[0].mean()

def calculate_zero_crossing_rate(y):
    """Calculates the zero crossing rate of the audio signal."""
    return librosa.feature.zero_crossing_rate(y)[0].mean()

def calculate_rms(y):
    """Calculates the root mean square (RMS) value of the audio signal."""
    return librosa.feature.rms(y=y)[0].mean()

def calculate_energy(y):
    """Calculates the energy of the audio signal."""
    return np.sum(y ** 2) / len(y)

def calculate_mfccs(y, sr, n_mfcc=13):
    """Calculates the Mel-Frequency Cepstral Coefficients (MFCCs) of the audio signal."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).mean(axis=1)

def extract_audio_features(file_path):
    """
    Extracts audio features from a given .wav file.

    Parameters:
        file_path (str): Path to the .wav file.

    Returns:
        dict: Dictionary containing pitch, ZCR, RMS, energy, and MFCCs.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Calculate features using individual functions
        pitch = calculate_pitch(y, sr)
        zcr = calculate_zero_crossing_rate(y)
        rms = calculate_rms(y)
        energy = calculate_energy(y)
        mfccs = calculate_mfccs(y, sr)

        return {
            "Pitch": pitch,
            "Zero Crossing Rate": zcr,
            "RMS": rms,
            "Energy": energy,
            **{f"MFCC_{i+1}": mfcc for i, mfcc in enumerate(mfccs)}
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_directory(directory_path, output_csv):
    """
    Processes all .wav files in a directory, extracts features, and saves them to a CSV file.

    Parameters:
        directory_path (str): Path to the directory containing .wav files.
        output_csv (str): Path to save the resulting CSV file.
    """
    data = []

    # Iterate over all files in the directory
    i = 0
    for file_name in os.listdir(directory_path):
        i +=1
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory_path, file_name)

            # Extract features
            features = extract_audio_features(file_path)

            if features:
                # Determine label based on the file name
                label = "Unknown"
                if "speech" in file_name.lower():
                    label = "Speech"
                elif "noise" in file_name.lower():
                    label = "Noise"
                elif "music" in file_name.lower():
                    label = "Music"

                # Append features and label
                features["Label"] = label
                features["File Name"] = file_name
                data.append(features)
                print(f"DONE {i}")


    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
    else:
        print("No .wav files found in the directory.")





def predict_sound_class(file_path, model, scaler, label_encoder):
    """
    Predict the class of a sound file (.wav) using a pre-trained MLP model.
    
    Parameters:
        file_path (str): Path to the .wav file.
        model (keras.Model): Loaded Keras model for prediction.
        scaler (StandardScaler): Pre-trained scaler for feature standardization.
        label_encoder (LabelEncoder): Pre-trained label encoder to decode class labels.
        
    Returns:
        str: Predicted class label.
    """

    
    # Extract features from the .wav file
    feature_dict = extract_audio_features(file_path)
    
    # Convert the dictionary into a NumPy array
    features = np.array(list(feature_dict.values())).reshape(1, -1)
    
    # Preprocess the features (e.g., standardization)
    features_scaled = scaler.transform(features)
    
    # Run the prediction
    prediction = model.predict(features_scaled)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Convert the index to the corresponding class label
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_class_label


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError(f"Critical frequencies must be 0 < Wn < 1. Got low: {low}, high: {high}.")
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def process_wav_to_memory(input_file, lowcut=100, highcut=12000, order=5):
    # Load the audio file
    data, sr = librosa.load(input_file, sr=None)  # Keep the original sample rate
    
    # Apply band-pass filter
    filtered_data = apply_bandpass_filter(data, lowcut, highcut, 48000, order)
    
    # Save to an in-memory buffer instead of a file
    buffer = BytesIO()
    sf.write(buffer, filtered_data, sr, format='WAV')
    buffer.seek(0)  # Reset buffer position
    
    return buffer

if __name__ == "__main__":
    # Load the trained model
    model = load_model("sound_classifier_mlp.h5")

    # Load the scaler
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load the label encoder
    with open("label_encoder.pkl", "rb") as label_encoder_file:
        label_encoder = pickle.load(label_encoder_file)

    # Input .wav file path (replace with your file path or take input from command line)
    if len(sys.argv) > 1:
        wav_file_path = sys.argv[1]
    else:
        wav_file_path = "example.wav"  # Default file path for testing

    # Predict the sound class
    wav_file_path = process_wav_to_memory(wav_file_path)
    try:
        predicted_class = predict_sound_class(wav_file_path, model, scaler, label_encoder)
        print(f"The predicted class for the audio file '{wav_file_path}' is: {predicted_class}")
    except Exception as e:
        print(f"An error occurred: {e}")
    