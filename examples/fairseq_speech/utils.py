import re
from collections import OrderedDict
from typing import Dict

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture, MODEL_REGISTRY
from fairseq.models.speech_to_text.xm_transformer import (
    base_architecture,
    XMTransformerModel,
)


TEST_WAITK_XMTF_ARCH_NAME = "simul_test_time_wait-k_xm_transformer"

if TEST_WAITK_XMTF_ARCH_NAME not in MODEL_REGISTRY:

    @register_model(TEST_WAITK_XMTF_ARCH_NAME)
    class SimulXMTransformerModel(XMTransformerModel):
        """
        This is a dummy class used for text-time wait-k model for offline xm_transformer
        """

        @staticmethod
        def add_args(parser):
            super(XMTransformerModel, XMTransformerModel).add_args(parser)
            parser.add_argument(
                "--train-monotonic-only",
                action="store_true",
                default=False,
                help="Only train monotonic attention",
            )

        @classmethod
        def build_decoder(cls, args, task, embed_tokens):
            tgt_dict = task.tgt_dict

            from examples.simultaneous_translation.models.transformer_monotonic_attention import (
                TransformerMonotonicDecoder,
            )

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


def rename_state_dict_test_time_waitk(
    state: Dict, waitk_lagging, fixed_predicision_ratio
):
    state["cfg"]["model"].load_pretrained_encoder_from = None
    state["cfg"]["model"].load_pretrained_decoder_from = None
    state["cfg"]["model"]._name = TEST_WAITK_XMTF_ARCH_NAME
    state["cfg"]["model"].arch = TEST_WAITK_XMTF_ARCH_NAME
    state["cfg"]["model"].simul_type = "waitk_fixed_pre_decision"
    state["cfg"]["model"].noise_type = None
    state["cfg"]["model"].noise_mean = None
    state["cfg"]["model"].noise_var = None
    state["cfg"]["model"].energy_bias_init = 0
    state["cfg"]["model"].energy_bias = False
    state["cfg"]["model"].waitk_lagging = waitk_lagging
    state["cfg"]["model"].fixed_pre_decision_type = "average"
    state["cfg"]["model"].fixed_pre_decision_ratio = fixed_predicision_ratio
    state["cfg"]["model"].fixed_pre_decision_pad_threshold = 0.3

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
