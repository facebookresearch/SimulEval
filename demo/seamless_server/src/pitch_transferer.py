import librosa
import psola

class Pitch_Transferer:
    
    def transfer(self, src_arr, tgt_arr):
        frame_length = 2048
        hop_length = frame_length // 4
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')
        sr = 16000

        f0, voiced_flag, voiced_probabilities = librosa.pyin(src_arr,
            frame_length=frame_length,
            hop_length=hop_length,
            sr=sr,
            fmin=fmin,
            fmax=fmax
        )
        new_arr = psola.vocode(tgt_arr, sample_rate=int(sr), target_pitch=f0, fmin=fmin, fmax=fmax)
        return new_arr