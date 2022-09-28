import os
import sys
import json
from typing import Optional, Dict, Tuple, List
from argparse import Namespace
from simuleval.agents import SpeechToSpeechAgent
from fairseq.data.encoders import build_bpe
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

sys.path.append(os.path.dirname(__file__))
from fairseq_test_waitk_s2t_agent import FairseqTestWaitKS2TAgent


class FairseqTestWaitKS2SAgent(SpeechToSpeechAgent, FairseqTestWaitKS2TAgent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args, process_id)
        self.load_tts()

    def process_write(self, prediction: str) -> str:
        self.states["target"].append(prediction)
        samples, fs = self.get_tts_output(prediction)
        samples = samples.cpu().tolist()
        return json.dumps({"samples": samples, "sample_rate": fs}).replace(" ", "")

    def load_tts(self):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.tts_generator = task.build_generator(models, cfg)
        self.tts_task = task
        self.tts_models = [model.to(self.device) for model in models]

    def get_tts_output(self, text: str) -> Tuple[List[float], int]:
        sample = TTSHubInterface.get_model_input(self.tts_task, text)
        for key in sample["net_input"].keys():
            if sample["net_input"][key] is not None:
                sample["net_input"][key] = sample["net_input"][key].to(self.device)

            wav, rate = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_models[0], self.tts_generator, sample
            )
            return wav, rate
