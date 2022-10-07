import os
import sys
import json
import torch
import simuleval
from typing import Optional, Dict, Tuple, List
from argparse import Namespace
from simuleval import DEFAULT_EOS
from simuleval.agents import SpeechToSpeechAgent
from fairseq.data.encoders import build_bpe
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from g2p_en import G2p

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples", "fairseq_speech"))
from fairseq_test_waitk_s2t_agent import FairseqTestWaitKS2TAgent


class FairseqTestWaitKS2SAgent(SpeechToSpeechAgent, FairseqTestWaitKS2TAgent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args, process_id)
        self.load_tts()
        self.g2p = G2p()

    def compute_phoneme_count(self, string: str) -> int:
        return len([x for x in self.g2p(string) if x != " "])

    def get_next_target_full_word(self, force_decode: bool = False) -> Optional[str]:
        possible_full_words = self.target_bpe.decode(
            self.states["target_subword_buffer"]
        )
        if force_decode:
            # Return decoded full word anyways
            return possible_full_words if len(possible_full_words) > 0 else None

        possible_full_words_list = possible_full_words.split()
        if len(possible_full_words_list) > 1:
            self.states["target_subword_buffer"] = possible_full_words_list[1]
            return possible_full_words_list[0]

        # Not ready yet
        return None

    def reset(self) -> None:
        super().reset()
        self.states["target_word_buffer"] = ""

    @staticmethod
    def add_args(parser):
        FairseqTestWaitKS2TAgent.add_args(parser)
        parser.add_argument(
            "--num-emit-phoneme",
            type=int,
            default=3,
            help="Minimal Number of the phonemes every write.",
        )

    def possible_write(self, decoder_states: torch.FloatTensor) -> None:
        log_probs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )
        index = log_probs.argmax(dim=-1)[0, 0].item()
        self.update_target(index)

        # Only write full word to server
        is_finished = index == self.model.decoder.dictionary.eos()
        if is_finished:
            self.finish_eval()

        possible_full_word = self.get_next_target_full_word(force_decode=is_finished)

        if possible_full_word is not None:
            self.states["target_word_buffer"] += " " + possible_full_word

        if not is_finished and (
            possible_full_word is None
            or self.compute_phoneme_count(self.states["target_word_buffer"])
            <= self.args.num_emit_phoneme
        ):
            # Not sure whether it's a full word now, just read more input

            self.read()
        else:
            if len(self.states["target_word_buffer"].strip()) > 0:
                # Send the sub
                self.write(self.states["target_word_buffer"])

    def process_write(self, prediction: str) -> str:
        if prediction == DEFAULT_EOS:
            return []
        self.states["target"] += self.states["target_word_buffer"].split()
        # print(self.states["target_word_buffer"])
        samples, fs = self.get_tts_output(self.states["target_word_buffer"])
        self.states["target_word_buffer"] = ""
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
