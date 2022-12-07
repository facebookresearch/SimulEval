import copy
import re
import math
from pathlib import Path
from collections import OrderedDict
from typing import Dict
from argparse import Namespace

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import fairseq
from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture, MODEL_REGISTRY
from fairseq.models.speech_to_text.xm_transformer import (
    base_architecture,
    XMTransformerModel,
)

from fairseq.models.speech_to_text.xm_transformer_unity import (
    build_embedding,
    xm_t_base_architecture,
    XMTransformerModelUnitY,
)

from fairseq.models.streaming.modules.monotonic_transformer_decoder import (
    TransformerMonotonicDecoder,
)

from simuleval.agents import Agent


TEST_WAITK_XMTF_ARCH_NAME = "simul_test_time_wait-k_xm_transformer"
TEST_WAITK_XMTF_UNITY_ARCH_NAME = "simul_test_time_wait-k_xm_transformer_t2"

if TEST_WAITK_XMTF_ARCH_NAME not in MODEL_REGISTRY:

    @register_model(TEST_WAITK_XMTF_ARCH_NAME)
    class SimulXMTransformerModel(XMTransformerModel):
        """
        This is a dummy class used for text-time wait-k model for offline xm_transformer
        """

        @classmethod
        def build_decoder(cls, args, task, embed_tokens):
            tgt_dict = task.tgt_dict

            decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

            if getattr(args, "load_pretrained_decoder_from", None):
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=args.load_pretrained_decoder_from
                )
            return decoder

    @register_model_architecture(
        model_name=TEST_WAITK_XMTF_ARCH_NAME, arch_name=TEST_WAITK_XMTF_ARCH_NAME
    )
    def simul_base_architecture(args):
        base_architecture(args)


if TEST_WAITK_XMTF_UNITY_ARCH_NAME not in MODEL_REGISTRY:

    @register_model(TEST_WAITK_XMTF_UNITY_ARCH_NAME)
    class SimulXMTransformerModelUnitY(XMTransformerModelUnitY):
        """
        This is a dummy class used for text-time wait-k model for offline xm_transformer
        """

        @classmethod
        def build_text_decoder(cls, args, tgt_dict):
            _args = copy.deepcopy(args)

            if args.adaptor_proj or args.encoder_proj:  # not V0 arch
                _args.encoder_embed_dim = _args.decoder_embed_dim
            _args.layerdrop = _args.decoder_layerdrop
            _args.decoder_layers = _args.translation_decoder_layers

            embed_tokens = build_embedding(tgt_dict, _args.decoder_embed_dim)

            for key in [
                "simul_type",
                "waitk_lagging",
                "waitk_consecutive_writes",
                "fixed_pre_decision_type",
                "fixed_pre_decision_ratio",
                "fixed_pre_decision_pad_threshold",
            ]:
                setattr(_args, key, getattr(args, key))

            decoder = TransformerMonotonicDecoder(_args, tgt_dict, embed_tokens)

            return decoder

    @register_model_architecture(
        model_name=TEST_WAITK_XMTF_UNITY_ARCH_NAME,
        arch_name=TEST_WAITK_XMTF_UNITY_ARCH_NAME,
    )
    def simul_base_architecture(args):
        xm_t_base_architecture(args)


def rename_state_dict_test_time_waitk(state: Dict, args: Namespace):
    state["cfg"]["model"].load_pretrained_encoder_from = None
    state["cfg"]["model"].load_pretrained_decoder_from = None
    state["cfg"]["model"]._name = (
        "simul_test_time_wait-k_" + state["cfg"]["model"]._name
    )
    state["cfg"]["model"].arch = "simul_test_time_wait-k_" + state["cfg"]["model"].arch
    state["cfg"]["model"].simul_type = args.simul_type
    state["cfg"]["model"].noise_type = None
    state["cfg"]["model"].noise_mean = None
    state["cfg"]["model"].noise_var = None
    state["cfg"]["model"].energy_bias_init = 0
    state["cfg"]["model"].energy_bias = False
    state["cfg"]["model"].waitk_lagging = args.waitk_lagging
    state["cfg"]["model"].waitk_consecutive_writes = args.waitk_consecutive_writes
    state["cfg"]["model"].fixed_pre_decision_type = "average"
    state["cfg"]["model"].fixed_pre_decision_ratio = args.fixed_predicision_ratio
    state["cfg"]["model"].fixed_pre_decision_pad_threshold = 0.3
    state["cfg"]["model"].target_letter_decoder_args = getattr(
        args, "target_letter_decoder_args", None
    )

    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if re.match(r"decoder\.layers\..\.encoder_attn", key):
            new_key = key.replace("k_proj", "k_proj_soft").replace(
                "q_proj", "q_proj_soft"
            )
            component_state_dict[new_key] = state["model"][key]
            component_state_dict[key] = state["model"][key]
        else:
            component_state_dict[key] = state["model"][key]
    return component_state_dict


def add_waitk_args(parser):
    parser.add_argument(
        "--simul-type",
        type=str,
        default="waitk_fixed_pre_decision",
        help="Type of wait-k",
    )
    parser.add_argument(
        "--fixed-predicision-ratio",
        type=int,
        default=8,
        help="The ratio of decision making every number of encoder states.",
    )
    parser.add_argument(
        "--waitk-lagging",
        type=int,
        required=True,
        help="Wait K lagging",
    )
    parser.add_argument(
        "--waitk-consecutive-writes",
        type=int,
        help="Wait K consecutive writes",
        default=1,
    )


def test_time_waitk_agent(agent: Agent):
    class TestTimeWaitKAgent(agent):
        @staticmethod
        def add_args(parser):
            super(agent, TestTimeWaitKAgent).add_args(parser)
            add_waitk_args(parser)
            try:
                # Try to add new add_args
                agent.add_args(parser)
            except:
                pass

        def process_checkpoint_state(self, state: Dict) -> Dict:
            return rename_state_dict_test_time_waitk(state, args=self.args)

    TestTimeWaitKAgent.__name__ = agent.__name__

    return TestTimeWaitKAgent


SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.frame_shift
        self.window_size = args.frame_length
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = (
            None
            if args.global_cmvn == ""
            else np.load(args.global_cmvn, allow_pickle=True)
        )

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # the number of frames used for feature extraction
        # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift :
        ]

        torch.manual_seed(1)
        output = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size,
        ).numpy()

        output = self.transform(output)

        return torch.from_numpy(output)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x
