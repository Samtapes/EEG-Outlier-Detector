import numpy as np

class EEGDataset:
    def __init__(self, data, datalabels, labels, artifact_label, seconds=0.5, fs=500.0, initial_time=200, channels=22):
        self.data = np.load(data)
        self.datalabels = np.load(datalabels)
        
        self.labels = labels
        self.artifact_label = artifact_label
        
        self.seconds = seconds
        self.fs = fs 
        self.duration = int(self.fs * self.seconds)
        self.initial_time = initial_time
        self.channels = channels
        
    def get_trials_from_channels(self):
        trials = []
        classes = []
        artifacts = []
        
        for i in range(self.data.shape[2]):
            class_ = self.datalabels[i]
            if class_ not in self.labels:
                continue
            # [channels x duration, trials]
            trial = self.data[:self.channels,self.initial_time:self.duration+self.initial_time,i]
            trials.append(trial)
            classes.append(class_)
            
            if class_ == self.artifact_label: artifacts.append(1)
            else: artifacts.append(0)

        trials = np.array(trials)
        # feature_arr = extractFeatures(trials)
                
        return trials, classes, artifacts