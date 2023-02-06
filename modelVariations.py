import numpy as np
from dataset import EEGDataset
from pyod.models import hbos
from pyod.utils.data import evaluate_print
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score, auc, roc_curve
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import pandas as pd

#####
# Trial or epoch = eeg segment
#####

channels = 32
selected_channels = 32
dataset_trial_duration = 1200 # Original dataset size of each trial (1.2 seconds)
fs = 1000.0        # Original dataset eeg sampling rate

seconds = [0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.600]    # Size of each segment we want
new_fs = 500.0     # Resize the dataset eeg sampling rate to 250.0 Hz
initial_time = 500 # We aren't going to pick the firsts 500ms because don't contain noise
save_model = False

# We want trials of 0.5 seconds or 500ms
# An trial array of EEG in 1000 Hz of 0.5s is size 500
# We will resample the data to 500 Hz and we want 0.5s for each trial
# So each trial array will have size of 250 (0.5s in 500 Hz)


print("Data contain 32 channels")
print("Using data with duration of:",seconds[0],"to",seconds[-1],"seconds")
print("Sampling Frequency:",fs,"Hz")
print("Our trial Length (seconds * fs) will be from:",fs*seconds[0], "to", fs*seconds[-1])
print("We will resample the data frequency to:", new_fs)
print("So our trial length will be from:",new_fs*seconds[0],"to",new_fs*seconds[-1])


## TRAIN DATASET ##
train_data_path = './data/clr_EEG32.npy'
train_datalabels_path = './data/clr_Labels.npy'
train_labels = [7,8,9,10,11,12,13]
train_artifact_label = 7

## TEST DATASET ##
test_data_path = './data/ori_EEG32.npy'
test_datalabels_path = './data/ori_Labels.npy'
test_labels = [1,2,3,4,5,6,14]
test_artifact_label = 14

def merge(data, labels):
  assert len(data) == len(labels)
  new_data = data.reshape(-1, data.shape[2])
  new_labels = [labels[0]] * data.shape[1]
  for i in range(1, len(labels)):
    new_data[data.shape[1]*i:data.shape[1]*(i+1)] = data[i, :, :]
    new_labels += [labels[i]] * data.shape[1]

  return np.array(new_data), np.array(new_labels)


## TESTING AND EVALUATING MULTIPLE MODELS
f1_scores = []
cohen_scores = []
roc_scores = []
traditional_scores = []

for i in range(len(seconds)):
  dataset = EEGDataset(train_data_path, train_datalabels_path, train_labels, train_artifact_label, seconds[i], fs, initial_time, selected_channels)
  train_eegData, train_classes, train_artifacts = dataset.get_trials_from_channels()

  test_dataset = EEGDataset(test_data_path, test_datalabels_path, test_labels, test_artifact_label, seconds[i], fs, initial_time, selected_channels)
  test_eegData, test_classes, test_artifacts = test_dataset.get_trials_from_channels()


  train_eegData, train_artifacts = merge(train_eegData, train_artifacts)
  test_eegData, test_artifacts = merge(test_eegData, test_artifacts)

  print("Segment duration:", seconds[i]*1000, " ms")
  # [chans * epochs x ms]

  print("="*20)
  print(f"RESAMPLING THE DATA {fs} Hz ({seconds[i]}s) to {new_fs} Hz")
  train_eegData = signal.resample(train_eegData, int(new_fs*seconds[i]), axis=1)
  test_eegData = signal.resample(test_eegData, int(new_fs*seconds[i]), axis=1)
  print("Train data new shape:",train_eegData.shape)
  print("Train data new labels shape:",train_artifacts.shape)



  #########
  ## TRAINING THE MODEL

  clf = hbos.HBOS(n_bins=17, alpha=0.07, tol=0.5,contamination=.15)
  clf.fit(train_eegData)

  print("="*20)
  print("TRAINING THE MODEL")

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

  accuracy = sum(pred == test_artifacts) / len(pred)
  print("Test Accuracy:",accuracy)
  evaluate_print(clf, test_artifacts, pred)
  print("\nConfusion Matrix")
  print(confusion_matrix(test_artifacts, pred, labels=[1,0]))

  f1 = f1_score(test_artifacts, pred)
  cohen = cohen_kappa_score(test_artifacts, pred)

  fpr, tpr, thresholds = roc_curve(y_true=test_artifacts, y_score=pred)
  auc_ = auc(fpr, tpr)

  print("Test Cohen Kappa:",cohen)
  print("Test F1 Score:",f1)

  f1_scores.append(f1)
  cohen_scores.append(cohen)
  roc_scores.append(auc_)
  traditional_scores.append(accuracy)


milliseconds = []
for i in seconds:
  milliseconds.append(i * 1000)


dict_ = {'F1_Score': f1_scores, 'Cohen_Kappa': cohen_scores, 'AUC_Score': roc_scores, 'Accuracy': traditional_scores, 'Time_ms': milliseconds}
df = pd.DataFrame(dict_)
df.to_csv('./accuracy/scores.csv')


plt.plot(milliseconds, f1_scores)
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("F1 Score")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/f1_scores.png")


plt.plot(milliseconds, cohen_scores, color='orange')
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("Cohen Kappa")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/cohen_kappas.png")

plt.plot(milliseconds, roc_scores, color='red')
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("AUC")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/auc_scores.png")

plt.plot(milliseconds, traditional_scores, color='green')
plt.xlabel("Milliseconds")
plt.ylabel("Score")
plt.title("Traditional Scores")
plt.grid(True)
plt.show()
plt.savefig("./accuracy/trad_scores.png")



## SAVING THE MODEL
if save_model:
  print("="*20)
  print("Saving the model...")
  filename = 'model.pickle'
  pickle.dump(clf, open(filename, 'wb'))