import whisper
import torch

from stable_whisper import load_model


class Translator:
    def __init__(self):
        # On Mac: Use 'cpu' and not 'mps' due to current Pytorch issue
        # with MPS https://github.com/pytorch/pytorch/issues/87886
        # self.model = load_model('medium', 'cpu')
        self.model = load_model('medium', 'cuda')
        
        # import pdb
        # pdb.set_trace()
        # self.model.device = torch.device("cuda")

    def transcribe(self, audio):
        return self.model.transcribe(audio)

    def translate(self, audio):
        return self.model.transcribe(audio, task="translate", suppress_silence=False)
