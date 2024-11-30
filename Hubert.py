import numpy as np
import torchaudio

def Hubert_Features_Extraction(signal):
    
    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert = bundle.get_model()

    wf, sr = torchaudio.load(signal)
    
    resample = torchaudio.transforms.Resample(sr, 16000)
    waveform = resample(wf)

    features, _ = hubert.extract_features(waveform)

    feature = features[11].detach().numpy()

    feature_time_mean = np.mean(feature, axis=1).reshape(feature.shape[2])
    return feature_time_mean

# Recordings for numbers 0-9 
train_files = ['zero_1.wav', 'one_1.wav', 'two_1.wav', 'three_1.wav', 'four_1.wav',
               'five_1.wav', 'six_1.wav', 'seven_1.wav', 'eight_1.wav', 'nine_1.wav']
test_files = ['zero_2.wav', 'one_2.wav', 'two_2.wav', 'three_2.wav', 'four_2.wav',
               'five_2.wav', 'six_2.wav', 'seven_2.wav', 'eight_2.wav', 'nine_2.wav']

train_hubert_features = [Hubert_Features_Extraction(train_file) for train_file in train_files]
True_Labels = list(range(10))
Predicted_Labels = []

for test_signal in test_files:
    test_hubert_feature = Hubert_Features_Extraction(test_signal)
    
    distance_labeld = [(np.sqrt(np.sum((train_hubert_features[i] - test_hubert_feature) ** 2)), i) for
                          i in range(len(train_hubert_features))]

    min_distance = min(distance_labeld, key=lambda x: x[0])
    Predicted_Labels.append(min_distance[1])

# Calculate recognition percentage
correct_predictions = sum(1 for true_label, predicted_label in zip(True_Labels, Predicted_Labels) if true_label == predicted_label)
total_samples = len(True_Labels)
recognition_percentage = (correct_predictions / total_samples) * 100

print(f"Recognition Percentage: {recognition_percentage:.1f}%")
