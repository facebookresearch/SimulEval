import io
import torch
class TTS_engine:

    def __init__(self):
        
        model_id = "v3_en"
        language = "en"
    
        device = torch.device('cuda')
        self.model, example_text = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=model_id
        )
        self.model.to(device)  # gpu or cpu

    def tts_gen(self, ssml_text, speaker="en_0", sr=24000):
        audio = self.model.apply_tts(
            ssml_text=ssml_text,
            speaker=speaker,
            sample_rate=sr
        )
        audio_arr = audio.detach().numpy()
        return audio_arr