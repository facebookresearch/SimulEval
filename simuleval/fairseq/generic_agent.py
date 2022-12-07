import os
import re
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, List
from fairseq import checkpoint_utils, tasks
from argparse import Namespace
from simuleval.agents import (
    Agent,
    SpeechToTextAgent,
    SpeechToSpeechAgent,
    TextToTextAgent,
)
from simuleval.agents.actions import ReadAction, WriteAction
from fairseq.utils import import_user_module
from simuleval import DEFAULT_EOS


class FairseqSimulAgent(Agent):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.fixed_predicision_ratio = args.fixed_predicision_ratio

        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.logger = logging.getLogger(f"simuleval.{type(self).__name__}")
        self.setup_device(args.device)

        logging.disable(logging.CRITICAL)
        self.load_checkpoint()
        logging.disable(logging.NOTSET)

    @property
    def source_segment_size(self):
        return self.fixed_predicision_ratio * 40

    def setup_device(self, device: List[str]):
        device = device[0]
        self.device = device
        if self.device != "cpu":
            try:
                torch.FloatTensor([1.0]).to(self.device)
                self.logger.info(
                    f"Using device: {self.device}. (process id {os.getpid()})"
                )
            except Exception as e:
                self.logger.error(f"Failed to use device: {self.device}, Error: {e}")
                self.logger.error(f"Change to CPU")
                self.device = "cpu"

    def reset(self):
        super().reset()
        self.is_finish_encoder_update = False
        self.states["encoder_states"] = None
        self.states["source_samples"] = []
        self.states["target_indices"] = []
        self.states["incremental_states"] = {}

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--checkpoint', type=str, required=True,
                            help='path to your pre-trained model.')
        parser.add_argument("--device", type=str, default="cuda:0", nargs="+",
                            help="Device to use")
        parser.add_argument("--init-target-token", default=None,
                            help="Init target token")
        parser.add_argument('--max-len-a', type=float, default=0.125,
                            help="Max length of predictions, a in ax + b")
        parser.add_argument('--max-len-b', type=float, default=10,
                            help="Max length of predictions, b in ax + b")
        try:
            parser.add_argument("--fairseq-data", type=str, default=None,
                                help="Path of fairseq data binary")
            parser.add_argument("--fairseq-config", type=str, default=None,
                                help="Path to fairseq config yaml file")
        except:
            pass
        # fmt: on
        return parser

    def process_checkpoint_state(self, state: Dict) -> Dict:
        return state

    def load_checkpoint(self):
        filename = self.args.checkpoint
        import_user_module(self.args)
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        task_args = state["cfg"]["task"]

        if self.args.fairseq_config is not None:
            task_args.config_yaml = self.args.fairseq_config

        if self.args.fairseq_data is not None:
            task_args.data = self.args.fairseq_data

        task = tasks.setup_task(task_args)
        self.fairseq_task = task

        component_state_dict = self.process_checkpoint_state(state)
        state["cfg"]["model"].max_positions = 1024
        state["cfg"]["model"].max_source_positions = 1024
        state["cfg"]["model"].max_target_positions = 1024
        state["cfg"]["model"].load_pretrained_decoder_from = None
        state["cfg"]["model"].w2v_path = state["cfg"]["model"].w2v_path.replace(
            "/large_experiments/ust", "/large_experiments/seamless/ust"
        )
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(component_state_dict, strict=False)
        self.model.eval()
        self.model.share_memory()
        self.model.to(self.device)

        if task.data_cfg.config.get("eos_token", None):
            self.init_target_index = self.model.decoder.dictionary.indices[
                task.data_cfg.config["eos_token"]
            ]
        else:
            self.init_target_index = self.model.decoder.dictionary.eos()

    @property
    def target_index_tensor(self) -> torch.LongTensor:
        return (
            torch.LongTensor([self.init_target_index] + self.states["target_indices"])
            .to(self.device)
            .unsqueeze(0)
        )

    def get_current_steps(
        self, source: torch.FloatTensor, target: torch.LongTensor
    ) -> Dict[str, int]:
        return {
            "src": source.size(0),
            "tgt": target.size(1),
        }

    @property
    def min_input_length(self):
        conv_layers = (
            self.model.encoder.w2v_encoder.w2v_model.feature_extractor.conv_layers
        )
        length = conv_layers[-1][0].kernel_size[0]
        for conv_layer in conv_layers:
            length *= conv_layer[0].stride[0]
        return length

    @property
    def source_length(self):
        if self.states["encoder_states"] is None:
            return 0
        return self.states["encoder_states"]["encoder_out"][0].size(0)

    @property
    def target_length(self):
        return len(self.states["target_indices"])

    @property
    def max_len(self):
        return self.max_len_a * self.source_length + self.max_len_b

    def generate_token_from_states(self, decoder_states):
        log_probs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )
        index = log_probs.argmax(dim=-1)[0, 0].item()

        if index != self.init_target_index:
            self.states["target_indices"].append(index)

        if index == self.model.decoder.dictionary.eos():
            return DEFAULT_EOS

        bpe_token = self.model.decoder.dictionary.string([index])

        if re.match(r"^\[.+_.+\]$", bpe_token) or len(bpe_token) == 0:
            # Language Indicator or UNK
            return None

        return bpe_token

    @torch.no_grad()
    def policy(self) -> None:
        if self.states["encoder_states"] is None:
            if not self.is_finish_read:
                return ReadAction()
            else:
                return WriteAction(DEFAULT_EOS)

        # 1. Prepare decoder input
        tgt_indices = self.target_index_tensor
        self.states["incremental_states"]["steps"] = self.get_current_steps(
            self.states["encoder_states"]["encoder_out"][0], tgt_indices
        )
        self.states["incremental_states"]["online"] = {
            "only": torch.tensor(not self.is_finish_read)
        }

        # 1.1 Run decoder
        decoder_states, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=self.states["encoder_states"],
            incremental_state=self.states["incremental_states"],
        )

        # 1.2. Make decision based on decoder output
        if outputs.action == 1 or self.is_finish_read:
            # 1.2.2 Predict
            token = self.generate_token_from_states(decoder_states)
            # 1.2.2 Check if finish then write
            if token != DEFAULT_EOS and self.target_length <= self.max_len:
                if token is not None:
                    return WriteAction(token)
            else:
                if self.is_finish_read:
                    return WriteAction(DEFAULT_EOS)
                else:
                    # Restart because of early stop
                    self.reset()

        return ReadAction()


class FairseqSimulSpeechInputAgent(FairseqSimulAgent):
    def update_encoder(self):
        # prepare encoder
        # This is an offline encoder so we update all the states every time
        # (Yes it's slow ...)
        try:
            torch.cuda.empty_cache()
            self.states["encoder_states"] = self.model.encoder(
                torch.FloatTensor(self.queue).to(self.device).unsqueeze(0),
                torch.LongTensor([len(self.queue)]).to(self.device),
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.reset()
            else:
                raise e

    @torch.no_grad()
    def push(self, source_info):
        self.queue += source_info["segment"]
        self.is_finish_read = source_info["finished"]
        if (
            not self.is_finish_encoder_update
            and len(self.queue) > self.min_input_length
        ):
            self.update_encoder()
            self.is_finish_encoder_update = self.is_finish_read


class FairseqSimulS2SAgent(FairseqSimulSpeechInputAgent, SpeechToSpeechAgent):
    pass


class FairseqSimulS2TAgent(FairseqSimulSpeechInputAgent, SpeechToTextAgent):
    pass


class FairseqSimulT2TAgent(FairseqSimulAgent, TextToTextAgent):
    pass
