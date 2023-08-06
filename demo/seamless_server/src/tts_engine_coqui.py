from TTS.api import TTS

class TTS_engine:

    def __init__(self):
        # print('available coqui TTS models:', TTS.list_models())
        model_name = TTS.list_models()[1]
        self.tts = TTS(model_name)

    
    def gen_arr(self, text):
        # print("tts speakers", self.tts.speakers)
        # print("tts languages", self.tts.languages)
        # import pdb
        # pdb.set_trace()
        speaker=None
        language=None
        if self.tts.is_multi_speaker:
            speaker = self.tts.speakers[0]

        if self.tts.is_multi_lingual:
            language =  self.tts.speakers[0]
        wav_arr = self.tts.tts("text", speaker=speaker, language=language)
        return wav_arr
