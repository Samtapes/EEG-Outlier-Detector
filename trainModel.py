import numpy as np
from dataset import EEGDataset
from pyod.models import hbos
from pyod.utils.data import evaluate_print
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
from scipy import signal
import pickle


#####
# Trial or epoch = eeg segment
#####

channels = 32
selected_channels = 32
dataset_trial_duration = 1200 # Original dataset size of each trial (1.2 seconds)
fs = 1000.0        # Original dataset eeg sampling rate
seconds = 0.300    # Size of each segment we want
new_fs = 500.0     # Resize the dataset eeg sampling rate to 250.0 Hz
initial_time = 500 # We aren't going to pick the firsts 500ms because don't contain noise
save_model = False

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


## TRAIN DATASET ##
train_data_path = './data/clr_EEG32.npy'
train_datalabels_path = './data/clr_Labels.npy'
train_labels = [7,8,9,10,11,12,13]
train_artifact_label = 7


dataset = EEGDataset(train_data_path, train_datalabels_path, train_labels, train_artifact_label, seconds, fs, initial_time, selected_channels)
train_eegData, train_classes, train_artifacts = dataset.get_trials_from_channels()


## TEST DATASET ##
test_data_path = './data/ori_EEG32.npy'
test_datalabels_path = './data/ori_Labels.npy'
test_labels = [1,2,3,4,5,6,14]
test_artifact_label = 14

test_dataset = EEGDataset(test_data_path, test_datalabels_path, test_labels, test_artifact_label, seconds, fs, initial_time, selected_channels)
test_eegData, test_classes, test_artifacts = test_dataset.get_trials_from_channels()




# eegData: 3D np array [epochs x chans x ms]
print("="*20)
print("Train data shape:",train_eegData.shape)
print("Test data shape:",test_eegData.shape)

print("Artifacts in train data:",sum(train_artifacts) / len(train_artifacts))
print("Artifacts in test data:",sum(test_artifacts) / len(test_artifacts))



######
## EXTRACTING FEATURES
# train_eegData = train_eegData.transpose(1,0,2)
# test_eegData = test_eegData.transpose(1,0,2)
# eegData shape needs to be: [chans x epochs x ms] 
# print("="*20)
# print("eeg data new shape:",train_eegData.shape)

# train_eegData = train_eegData[:2, :int(train_eegData.shape[1] / 2), :]
# test_eegData = test_eegData[:2, :int(test_eegData.shape[1] / 2), :]
# print("eeg data new shape:",train_eegData.shape)

# Joining channels in epochs
# new shape [chans * epochs x ms]


def merge(data, labels):
  assert len(data) == len(labels)
  new_data = data.reshape(-1, data.shape[2])
  new_labels = [labels[0]] * data.shape[1]
  for i in range(1, len(labels)):
    new_data[data.shape[1]*i:data.shape[1]*(i+1)] = data[i, :, :]
    new_labels += [labels[i]] * data.shape[1]

  return np.array(new_data), np.array(new_labels)

train_eegData, train_artifacts = merge(train_eegData, train_artifacts)
test_eegData, test_artifacts = merge(test_eegData, test_artifacts)

print("="*20)
print("MERGING CHANNELS WITH EPOCHS")
print("Train data new shape:",train_eegData.shape)
print("Train data new labels shape:",train_artifacts.shape)
# [chans * epochs x ms]

print("="*20)
print(f"RESAMPLING THE DATA {fs} Hz ({seconds}s) to {new_fs} Hz")
train_eegData = signal.resample(train_eegData, int(new_fs*seconds), axis=1)
test_eegData = signal.resample(test_eegData, int(new_fs*seconds), axis=1)
print("Train data new shape:",train_eegData.shape)
print("Train data new labels shape:",train_artifacts.shape)



#########
## TRAINING THE MODEL

clf = hbos.HBOS(n_bins=17, alpha=0.07, tol=0.5,contamination=.15)
clf.fit(train_eegData)

print("="*20)
print("TRAINING THE MODEL")

print("Labels length discovered by the model:",len(clf.labels_))

print("Model Accuracy:",sum(clf.labels_ == train_artifacts) / len(clf.labels_))

evaluate_print(clf, train_artifacts, clf.labels_)

print("\nConfusion Matrix")
print(confusion_matrix(train_artifacts, clf.labels_, labels=[1,0]))

print("\nCohen kappa:",cohen_kappa_score(train_artifacts, clf.labels_))
print("F1 Score:",f1_score(train_artifacts, clf.labels_))


## EVALUATING THE MODEL ##
pred = clf.predict(test_eegData)

print("="*20)
print("EVALUATING THE MODEL")
print("Test Accuracy:",sum(pred == test_artifacts) / len(pred))
evaluate_print(clf, test_artifacts, pred)
print("\nConfusion Matrix")
print(confusion_matrix(test_artifacts, pred, labels=[1,0]))
print("Test Cohen Kappa:",cohen_kappa_score(test_artifacts, pred))
print("Test F1 Score:",f1_score(test_artifacts, pred))



## SAVING THE MODEL
if save_model:
  print("="*20)
  print("Saving the data...")
  filename = 'model.pickle'
  pickle.dump(clf, open(filename, 'wb'))