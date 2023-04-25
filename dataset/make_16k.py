import os
import sys
from pathlib import Path

import librosa
import soundfile as sf

Path('Audio_16k').mkdir(exist_ok=True)
CREMAD_DIR = Path(sys.argv[1])
print("Downsampling CREMA-D to 16k")
current_dir = CREMAD_DIR
for x, full_audio_name in enumerate(current_dir.rglob('*.wav')):
    print(str(x) + " downsampled")
    audio, sr = librosa.load(str(full_audio_name), sr=None)
    audio_name = full_audio_name.name
    assert sr == 16000
    sf.write(os.path.join('Audio_16k', audio_name), audio, 16000)
