import os
import numpy as np
import json
import random
from sklearn.model_selection import train_test_split
from pathlib import Path

Path('labels_sess').mkdir(exist_ok=True)
with open('metalabel.json', 'r') as f:
    metalabel = json.load(f)

labeldict = {
    'ANG': 'anger',
    'HAP': 'happy',
    'DIS': 'disgust',
    'SAD': 'sad',
    'FEA': 'fear',
    'NEU': 'neutral'
}

labels = {
    'Train': {},
    'Val': {},
    'Test': {}
}

audio_list, label_list = [], []
for audio in os.listdir('Audio_16k'):
    label_key = metalabel[audio]
    if label_key not in labeldict:
        continue
    audio_list.append(audio)
    label_list.append(labeldict[label_key])

X_train, X_rem, y_train, y_rem = train_test_split(
    audio_list, label_list,
    train_size=0.8,
    random_state=23,
    stratify=label_list)

# Now since we want the valid and test size to be equal (10% each of overall data).
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem,
    test_size=0.5,
    stratify=y_rem)

for i in range(len(X_train)):
    labels['Train'][X_train[i]] = y_train[i]

for i in range(len(X_valid)):
    labels['Val'][X_valid[i]] = y_valid[i]

for i in range(len(X_test)):
    labels['Test'][X_test[i]] = y_test[i]

with open(f'labels.json', 'w') as f:
    json.dump(labels, f, indent=4)
