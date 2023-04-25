import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

CREMAD_DIR = Path(sys.argv[1])

print ('Generating labels and train/validation/test groups...')
label_dir = Path('Audio_16k')

labeldict = {
    'ANG': 'anger',
    'HAP': 'happy',
    'DIS': 'disgust',
    'SAD': 'sad',
    'FEA': 'fear',
    'NEU': 'neutral'
}
audio_list, label_list = [], []

for x, full_audio_name in enumerate(label_dir.rglob('*.wav')):
    file_name = os.path.basename(full_audio_name).split('/')[-1]
    label = str(file_name)[-10:-7]
    if label not in labeldict:
        continue
    audio_list.append(file_name)
    label_list.append(labeldict[label])


labels = {
    'Train': {},
    'Val': {},
    'Test': {}
}

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