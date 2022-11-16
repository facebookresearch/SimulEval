import os
import sys
import json
import torch
import simuleval
from typing import Optional, Tuple, List
from argparse import Namespace
from simuleval.agents import SpeechToSpeechAgent
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from g2p_en import G2p

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples"))
from fairseq_speech.generic_agent import FairseqSimulS2SAgent
from fairseq_speech.utils import test_time_waitk_agent
from simuleval.postprocessor import GenericPostProcessor, SPMPostProcessor


class Fastspeech2PostProcessor(GenericPostProcessor):
    def __init__(
        self,
        spm_postprocessor: SPMPostProcessor,
        min_phoneme: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.spm_postprocessor = spm_postprocessor
        self.min_phoneme = min_phoneme
        self.device = torch.device(device)
        self.load_tts()
        self.g2p = G2p()
        self.is_finish = False

    def reset(self):
        super().reset()
        self.is_finish = False

    def load_tts(self):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.tts_generator = task.build_generator(models, cfg)
        self.tts_task = task
        self.tts_models = [model.to(self.device) for model in models]

    def compute_phoneme_count(self, string: str) -> int:
        return len([x for x in self.g2p(string) if x != " "])

    def get_tts_output(self, text: str) -> Tuple[List[float], int]:
        sample = TTSHubInterface.get_model_input(self.tts_task, text)
        for key in sample["net_input"].keys():
            if sample["net_input"][key] is not None:
                sample["net_input"][key] = sample["net_input"][key].to(self.device)

            wav, rate = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_models[0], self.tts_generator, sample
            )
            return wav, rate

    def push(self, item):
        # Only push full words
        self.spm_postprocessor.push(item)
        output = self.spm_postprocessor.pop()
        for o in output:
            if o is not None:
                self.deque.append(o)

    def pop(self):
        current_phoneme_counts = self.compute_phoneme_count(" ".join(self.deque))
        if current_phoneme_counts >= self.min_phoneme or self.is_finish:
            samples, fs = self.get_tts_output(" ".join(self.deque))
            samples = samples.cpu().tolist()
            self.reset()
            return json.dumps({"samples": samples, "sample_rate": fs}).replace(" ", "")


@test_time_waitk_agent
class FairseqTestWaitKS2SAgent(FairseqSimulS2SAgent):
    def build_postprocessor(self, args: Namespace) -> GenericPostProcessor:
        spm_postprocessor = super().build_postprocessor(args)
        return Fastspeech2PostProcessor(
            spm_postprocessor, args.num_emit_phoneme, args.device[0]
        )

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--num-emit-phoneme",
            type=int,
            default=3,
            help="Minimal Number of the phonemes every write.",
        )
