import os
import sys
import torch
import simuleval
from typing import Dict

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples", "fairseq_speech"))

from fairseq_generic_speech_agent import FairseqSimulS2TAgent
from utils import rename_state_dict_test_time_waitk


class FairseqTestWaitKS2TAgent(FairseqSimulS2TAgent):
    """
    Test-time Wait-K agent for speech-to-text translation.
    This agent load an offline model and run Wait-K policy
    """

    @staticmethod
    def add_args(parser):
        super(FairseqTestWaitKS2TAgent, FairseqTestWaitKS2TAgent).add_args(parser)
        parser.add_argument(
            "--fixed-predicision-ratio",
            type=int,
            default=3,
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

    def process_checkpoint_state(self, state: Dict) -> Dict:
        # Rename parameters to enable offline model in online decoding
        return rename_state_dict_test_time_waitk(
            state,
            self.args.waitk_lagging,
            self.args.fixed_predicision_ratio,
            self.args.waitk_consecutive_writes,
        )

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
            # 1.2.2 Check if finish then write
            if not self.check_finish_eval():
                self.write(token)
