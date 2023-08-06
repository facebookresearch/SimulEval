import numpy as np
import librosa

def convert_wav_to_np(audio_file, sr=16000) -> np.ndarray:
    audio_arr, _ = librosa.load(audio_file, sr = sr)
    return audio_arr
