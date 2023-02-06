import numpy as np
import matplotlib.pyplot as plt
from dataset import EEGDataset
from pyod import models
from pyod.models import hbos
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
from EEGExtract import extractFeatures

seconds = 3
fs = 250.0

print("Data contain 32 channels")
print("Using data with duration of:",seconds,"seconds")
print("Sampling Frequency:",fs,"Hz")
print("Trial Length (seconds * fs) =",seconds*fs)


## TRAIN DATASET ##
train_data_path = './data/clr_EEG32.npy'
train_datalabels_path = './data/clr_Labels.npy'
train_labels = [7,8,9,10,11,12,13]
train_artifact_label = 7

dataset = EEGDataset(train_data_path, train_datalabels_path, train_labels, train_artifact_label, seconds, fs)
train_eegData, train_classes, train_artifacts = dataset.get_trials_from_channels()


## TEST DATASET ##
test_data_path = './data/ori_EEG32.npy'
test_datalabels_path = './data/ori_Labels.npy'
test_labels = [1,2,3,4,5,6,14]
test_artifact_label = 14

test_dataset = EEGDataset(test_data_path, test_datalabels_path, test_labels, test_artifact_label, seconds, fs)
test_eegData, test_classes, test_artifacts = test_dataset.get_trials_from_channels()




# eegData: 3D np array [epochs x chans x ms]
print("="*20)
print("Train data shape:",train_eegData.shape)
print("Test data shape:",test_eegData.shape)

print("Artifacts in train data:",sum(train_artifacts) / len(train_artifacts))
print("Artifacts in test data:",sum(test_artifacts) / len(test_artifacts))



######
## EXTRACTING FEATURES
train_eegData = train_eegData.transpose(1,0,2)
test_eegData = test_eegData.transpose(1,0,2)
# eegData shape needs to be: [chans x epochs x ms] 
print("="*20)
print("eeg data new shape:",train_eegData.shape)

train_eegData = train_eegData[:2, :int(train_eegData.shape[1] / 6), :]
test_eegData = test_eegData[:2, :int(test_eegData.shape[1] / 6), :]
print("eeg data new shape:",train_eegData.shape)


##########
# Extract the tsalis Entropy
def shannonEntropy(eegData, bin_min, bin_max, binWidth):
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            counts, binCenters = np.histogram(eegData[chan,:,epoch], bins=np.arange(bin_min+1, bin_max, binWidth))
            nz = counts > 0
            prob = counts[nz] / np.sum(counts[nz])
            H[chan, epoch] = -np.dot(prob, np.log2(prob/binWidth))
    return H


ShannonRes = shannonEntropy(train_eegData, bin_min=-200, bin_max=200, binWidth=2)
print(ShannonRes.shape)



def shannonEntropy2(eegData, bin_min, bin_max, binWidth):
    print(eegData[0,:,0].shape)
    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    counts, binCenters = np.histogram(eegData, bins=np.arange(bin_min+1, bin_max, binWidth))
    print(counts.shape)
    nz = counts > 0
    prob = counts[nz] / np.sum(counts[nz])
    H = np.log2(prob/binWidth)
    return H

ShannonRes2 = shannonEntropy2(train_eegData, bin_min=-200, bin_max=200, binWidth=2)
print(ShannonRes2.shape)