import os
import re
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Optional
from fairseq import checkpoint_utils, tasks
from argparse import Namespace
from simuleval.agents import Agent, SpeechToTextAgent, SpeechToSpeechAgent
from simuleval.postprocessor import SPMPostProcessor
from fairseq.utils import import_user_module


class FairseqSimulAgent(Agent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args)
        self.source_segment_size = args.source_segment_size
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.logger = logging.getLogger(f"simuleval.{type(self).__name__}")
        if process_id is None:
            process_id = 0
        assert process_id <= len(args.device) or args.device == ["cpu"]

        if args.device == ["cpu"]:
            self.set_device("cpu")
        else:
            self.set_device(args.device[process_id])

        logging.disable(logging.CRITICAL)
        self.load_checkpoint()
        logging.disable(logging.NOTSET)

        if args.init_target_token:
            self.init_target_index = self.model.decoder.dictionary.indices[
                args.init_target_token
            ]
        else:
            self.init_target_index = self.model.decoder.dictionary.eos()

    def set_device(self, device: str) -> None:
        self.device = device
        try:
            torch.FloatTensor([1.0]).to(self.device)
            self.logger.info(f"Using device: {self.device}. (process id {os.getpid()})")
        except Exception as e:
            self.logger.error(f"Failed to use device: {self.device}, Error:")
            self.logger.error(f"Change to CPU")
            self.device = "cpu"

    def reset(self):
        super().reset()
        self.states["encoder_states"] = None
        self.states["source_samples"] = []
        self.states["target_indices"] = []
        self.states["target_subword"] = []
        self.states["incremental_states"] = {}

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--checkpoint', type=str, required=True,
                            help='path to your pre-trained model.')
        parser.add_argument("--fairseq-data", type=str, default=None,
                            help="Path of fairseq data binary")
        parser.add_argument("--fairseq-config", type=str, default=None,
                            help="Path to fairseq config yaml file")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothesis if the source is not finished")
        parser.add_argument("--device", type=str, default="cuda:0", nargs="+",
                            help="Device to use")
        parser.add_argument("--source-segment-size", type=int, default=200,
                            help="Source segment size in ms")
        parser.add_argument("--init-target-token", default=None,
                            help="Init target token")
        parser.add_argument('--max-len-a', type=float, default=0,
                            help="Max length of predictions, a in ax + b")
        parser.add_argument('--max-len-b', type=float, default=200,
                            help="Max length of predictions, b in ax + b")
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

    def eval(self, **kwargs) -> None:
        with torch.no_grad():
            super().eval(**kwargs)

    def get_target_index_tensor(self) -> torch.LongTensor:
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
        return self.states["encoder_states"]["encoder_out"][0].size(0)

    @property
    def target_length(self):
        return len(self.states["target_indices"])

    @property
    def max_len(self):
        return self.max_len_a * self.source_length + self.max_len_b

    def check_finish_eval(self):
        if (
            self.states["target_indices"][-1] == self.model.decoder.dictionary.eos()
            or self.target_length > self.max_len
        ):
            is_finished = True
        else:
            is_finished = False

        if is_finished:
            if self.is_finish_read or self.target_length > self.max_len:
                self.finish_eval()
            else:
                keep_index = self.index
                self.reset()
                self.index = keep_index
            return True

        return False

    def get_token_from_states(self, decoder_states):
        log_probs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )
        index = log_probs.argmax(dim=-1)[0, 0].item()

        if index != self.init_target_index:
            self.states["target_indices"].append(index)

        bpe_token = self.model.decoder.dictionary.string([index])

        if re.match(r"^\[.+_.+\]$", bpe_token) or len(bpe_token) == 0:
            # Language Indicator or UNK
            return None

        return bpe_token

    def policy(self) -> None:
        # 0.0 Read at the beginning
        while self.states["encoder_states"] is None:
            if self.is_finish_read:
                self.finish_eval()
                return
            self.read()

        # 0.1 Prepare decoder input
        tgt_indices = self.get_target_index_tensor()
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

        # 1.2. Make decision on model output
        if outputs.action == 0 and not self.is_finish_read:
            # 1.2.1 Read
            self.read()
        else:
            # 1.2.2 Predict
            token = self.get_token_from_states(decoder_states)
            self.states["target_subword"].append(token)
            # 1.2.2 Check if finish then write
            if not self.check_finish_eval():
                self.write(token)


class FairseqSimulSpeechInputAgent(FairseqSimulAgent):
    def build_postprocessor(self, args: Namespace):
        return SPMPostProcessor.from_fairseq_s2t_config(
            Path(args.fairseq_data) / args.fairseq_config
        )

    def process_read(self, source_info: Dict) -> Dict:
        self.states["source_samples"] += source_info["segment"]
        self.is_finish_read = source_info["finished"]
        torch.cuda.empty_cache()
        if len(self.states["source_samples"]) >= self.min_input_length:
            self.states["encoder_states"] = self.model.encoder(
                torch.FloatTensor(self.states["source_samples"])
                .to(self.device)
                .unsqueeze(0),
                torch.LongTensor([len(self.states["source_samples"])]).to(self.device),
            )


class FairseqSimulS2SAgent(FairseqSimulSpeechInputAgent, SpeechToSpeechAgent):
    pass

class FairseqSimulS2TAgent(FairseqSimulSpeechInputAgent, SpeechToTextAgent):
    pass
