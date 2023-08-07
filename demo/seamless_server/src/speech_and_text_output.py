# Provides a container to return both speech and text output from our model at the same time


class SpeechAndTextOutput:
    def __init__(
        self,
        text: str = None,
        speech_samples: list = None,
        speech_sample_rate: float = None,
        final: bool = False,
    ):
        self.text = text
        self.speech_samples = speech_samples
        self.speech_sample_rate = speech_sample_rate
        self.final = final
