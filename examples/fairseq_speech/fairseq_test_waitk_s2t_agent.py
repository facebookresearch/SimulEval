import os
import sys
import torch
from pathlib import Path
from typing import Optional, Dict
from argparse import Namespace
from simuleval.agents import SpeechToTextAgent
from fairseq.data.encoders import build_bpe
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig

sys.path.append(os.path.dirname(__file__))
from fairseq_generic_speech_agent import FairseqTestWaitKAgent


class FairseqTestWaitKS2TAgent(FairseqTestWaitKAgent, SpeechToTextAgent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args, process_id)
        self.target_bpe = build_bpe(
            Namespace(
                **S2TDataConfig(
                    Path(self.args.fairseq_data) / self.args.fairseq_config
                ).bpe_tokenizer
            )
        )

    def process_write(self, prediction: str) -> str:
        self.states["target"].append(prediction)
        return prediction

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

        if possible_full_word is None:
            # Not sure whether it's a full word now, just read more input
            self.read()
        else:
            # Send the sub
            self.write(possible_full_word)

    def policy(self) -> None:
        # 0.0 Read at the beginning
        while self.states["encoder_states"] is None:
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
            # 1.2.2 Write, but it's possible to skip base on current buffer
            self.possible_write(decoder_states)
