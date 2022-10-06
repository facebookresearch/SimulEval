import os
import sys
import re
import json
import torch
import logging
from typing import Dict, Optional
from fairseq import checkpoint_utils, tasks
from argparse import Namespace
from simuleval.agents import Agent

sys.path.append(os.path.dirname(__file__))
from utils import rename_state_dict_test_time_waitk


class FairseqSimulAgent(Agent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args)
        self.source_segment_size = args.source_segment_size
        self.logger = logging.getLogger(f"simuleval.{type(self).__name__}")
        if process_id is None:
            process_id = 0
        assert process_id <= len(args.device) or args.device == ["cpu"]

        if args.device == ["cpu"]:
            self.set_device("cpu")
        else:
            self.set_device(args.device[process_id])

        self.load_checkpoint()

        if args.init_target_token:
            self.init_target_index = self.model.decoder.dictionary.indices[args.init_target_token]
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
        self.states["target_subword_buffer"] = ""
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
        # fmt: on
        return parser

    def process_checkpoint_state(self, state: Dict) -> Dict:
        return state

    def load_checkpoint(self):
        filename = self.args.checkpoint
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]

        if self.args.fairseq_config is not None:
            task_args.config_yaml = self.args.fairseq_config

        if self.args.fairseq_data is not None:
            task_args.data = self.args.fairseq_data

        task = tasks.setup_task(task_args)

        component_state_dict = self.process_checkpoint_state(state)
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
            torch.LongTensor(
                [self.init_target_index] + self.states["target_indices"]
            )
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

    def process_read(self, source_info: Dict) -> Dict:
        self.states["source"].append(source_info)
        self.states["source_samples"] += source_info["segment"]
        self.is_finish_read = source_info["finished"]
        torch.cuda.empty_cache()
        self.states["encoder_states"] = self.model.encoder(
            torch.FloatTensor(self.states["source_samples"])
            .to(self.device)
            .unsqueeze(0),
            torch.LongTensor([len(self.states["source_samples"])]).to(self.device),
        )
        return source_info

    def process_write(self, prediction: str) -> str:
        self.states["target"].append(prediction)
        samples, fs = self.get_tts_output(prediction)
        samples = samples.cpu().tolist()
        return json.dumps({"samples": samples, "sample_rate": fs}).replace(" ", "")

    def update_target(self, pred_index):
        self.states["target_indices"].append(pred_index)
        bpe_token = self.model.decoder.dictionary.string([pred_index])

        if re.match(r"^\[.+_.+\]$", bpe_token):
            # Language Indicator
            return

        self.states["target_subword_buffer"] += " " + bpe_token


class FairseqTestWaitKAgent(FairseqSimulAgent):
    def process_checkpoint_state(self, state: Dict) -> Dict:
        return rename_state_dict_test_time_waitk(
            state, self.args.waitk_lagging, self.args.fixed_predicision_ratio
        )

    @staticmethod
    def add_args(parser):
        super(FairseqTestWaitKAgent, FairseqTestWaitKAgent).add_args(parser)
        parser.add_argument(
            "--fixed-predicision-ratio",
            type=int,
            default=3,
            help="Acoustic feature dimension.",
        )
        parser.add_argument(
            "--waitk-lagging",
            type=int,
            required=True,
            help="Acoustic feature dimension.",
        )
