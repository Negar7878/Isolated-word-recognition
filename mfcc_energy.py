import numpy as np
import librosa
import soundfile as sf

# Function to extract a fixed number of mel cepstral coefficients from a speech signal for each frame
def extract_mel_cepstrum_and_energy_features(signal, sr, frame_size_ms=30, frame_shift_ms=10, num_ceps=12):
    frame_size_sample = int((frame_size_ms / 1000) * sr)
    frame_shift_sample = int((frame_shift_ms / 1000) * sr)
    
    # Create frames with overlap
    frames = np.array([signal[i:i + frame_size_sample] for i in range(0, len(signal) - frame_size_sample + 1, frame_shift_sample)])
    
    # Apply Hamming window to each frame
    hamming_window = np.hamming(frame_size_sample)
    frames *= hamming_window
    
    # Calculate energy for each frame
    energy = np.sum(frames**2, axis=1)
    
    # Calculate mel cepstral coefficients
    mel_cepstral = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_ceps, hop_length=frame_shift_sample, n_fft=frame_size_sample)
    
    # Concatenating the features
    return np.c_[mel_cepstral.T[:len(energy), :], energy]

# Recordings for numbers 0-9 
train_files = ['zero_1.wav', 'one_1.wav', 'two_1.wav', 'three_1.wav', 'four_1.wav',
               'five_1.wav', 'six_1.wav', 'seven_1.wav', 'eight_1.wav', 'nine_1.wav']
test_files = ['zero_2.wav', 'one_2.wav', 'two_2.wav', 'three_2.wav', 'four_2.wav',
               'five_2.wav', 'six_2.wav', 'seven_2.wav', 'eight_2.wav', 'nine_2.wav']

train_signals = [librosa.resample(librosa.load(train_file)[0], orig_sr=librosa.load(train_file)[1], target_sr=16000) for train_file in train_files]
train_mfccs_energies = [extract_mel_cepstrum_and_energy_features(train_signal, 16000) for train_signal in train_signals]
True_Labels = list(range(10))
Predicted_Labels = []

for test_file in test_files:
    signal_test, sr_test = librosa.load(test_file, sr=None)
    resampled_signal_test = librosa.resample(signal_test, orig_sr=sr_test, target_sr=16000)
    mfcc_energy_test = extract_mel_cepstrum_and_energy_features(resampled_signal_test, sr=16000)
    
    D_ceps_labeld = [(np.sqrt(np.sum((train_mfccs_energies[i][:min_frame, :] - mfcc_energy_test[:min_frame, :]) ** 2)), i) for
                          i, min_frame in enumerate(map(lambda x: min(x.shape[0], mfcc_energy_test.shape[0]), train_mfccs_energies))]

    min_D_ceps = min(D_ceps_labeld, key=lambda x: x[0])
    Predicted_Labels.append(min_D_ceps[1])

# Calculate recognition percentage
correct_predictions = sum(1 for true_label, predicted_label in zip(True_Labels, Predicted_Labels) if true_label == predicted_label)
total_samples = len(True_Labels)
recognition_percentage = (correct_predictions / total_samples) * 100

print(f"Recognition Percentage: {recognition_percentage:.1f}%")
