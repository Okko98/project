import json
import os
import sys
from pathlib import Path

import numpy as np

CREMAD_DIR = Path(sys.argv[1])

print ('Generating metalabels...')
metalabel = {}
label_dir = Path('Audio_16k')
for x, full_audio_name in enumerate(label_dir.rglob('*.wav')):
    file_name = os.path.basename(full_audio_name).split('/')[-1]
    label = str(file_name)[-10:-7]
    metalabel[str(file_name)] = label
with open(f'metalabel.json', 'w') as f:
    json.dump(metalabel, f, indent=4)