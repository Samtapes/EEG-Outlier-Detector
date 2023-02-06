import numpy as np
from dataset import EEGDataset
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
from scipy import signal
import pickle
# from HBOS import HBOS

#####
# Trial or epoch = eeg segment
#####

channels = 32
selected_channels = 32
dataset_trial_duration = 1200 # Original dataset size of each trial (1.2 seconds)
fs = 1000.0        # Original dataset eeg sampling rate
seconds = 0.300      # Size of each segment we want
new_fs = 500.0     # Resize the dataset eeg sampling rate to 250.0 Hz
initial_time = 500 # We aren't going to pick the firsts 500ms because don't contain noise

# We want trials of 0.5 seconds or 500ms
# An trial array of EEG in 1000 Hz of 0.5s is size 500
# We will resample the data to 500 Hz and we want 0.5s for each trial
# So each trial array will have size of 250 (0.5s in 500 Hz)


print("Data contain 32 channels")
print("Using data with duration of:",seconds,"seconds")
print("Sampling Frequency:",fs,"Hz")
print("Our trial Length (seconds * fs) =",fs*seconds)
print("We will resample the data frequency to:", new_fs)
print("So our trial length will be:",new_fs*seconds)


## TEST DATASET ##
test_data_path = './data/ori_EEG32.npy'
test_datalabels_path = './data/ori_Labels.npy'
test_labels = [1,2,3,4,5,6,14]
test_artifact_label = 14

test_dataset = EEGDataset(test_data_path, test_datalabels_path, test_labels, test_artifact_label, seconds, fs, initial_time, selected_channels)
test_eegData, test_classes, test_artifacts = test_dataset.get_trials_from_channels()




# eegData: 3D np array [epochs x chans x ms]
print("="*20)
print("Test data shape:",test_eegData.shape)

print("Artifacts in test data:",sum(test_artifacts) / len(test_artifacts))



def merge(data, labels):
  assert len(data) == len(labels)
  new_data = data.reshape(-1, data.shape[2])
  new_labels = [labels[0]] * data.shape[1]
  for i in range(1, len(labels)):
    new_data[data.shape[1]*i:data.shape[1]*(i+1)] = data[i, :, :]
    new_labels += [labels[i]] * data.shape[1]

  return np.array(new_data), np.array(new_labels)

test_eegData, test_artifacts = merge(test_eegData, test_artifacts)

print("="*20)
print("MERGING CHANNELS WITH EPOCHS")
print("Test data new shape:",test_eegData.shape)
print("Test data new labels shape:",test_artifacts.shape)
# [chans * epochs x ms]

print("="*20)
print(f"RESAMPLING THE DATA {fs} Hz ({seconds}s) to {new_fs} Hz")
test_eegData = signal.resample(test_eegData, int(new_fs*seconds), axis=1)
print("Test data new shape:",test_eegData.shape)
print("Test data labels shape:",test_artifacts.shape)



## EVALUATING THE MODEL ##
clf = pickle.load(open("model.pickle", 'rb'))
pred = clf.predict(test_eegData)

print("="*20)
print("EVALUATING THE MODEL")
print("Test Accuracy:",sum(pred == test_artifacts) / len(pred))
print("\nConfusion Matrix")
print(confusion_matrix(test_artifacts, pred, labels=[1,0]))
print("Test Cohen Kappa:",cohen_kappa_score(test_artifacts, pred))
print("Test F1 Score:",f1_score(test_artifacts, pred))