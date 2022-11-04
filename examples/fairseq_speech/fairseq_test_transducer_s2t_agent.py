from gc import is_finalized
import os
import math
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import simuleval
from pathlib import Path
from typing import Optional, Dict
from argparse import Namespace
import torch.nn.functional as F
from simuleval.agents import SpeechToTextAgent
from fairseq.data.encoders import build_bpe
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq import checkpoint_utils, tasks
from fairseq.utils import import_user_module

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples", "fairseq_speech"))
from fairseq_generic_speech_agent import FairseqSimulS2TAgent
from utils import OnlineFeatureExtractor

class FairseqTestTransducerS2TAgent(FairseqSimulS2TAgent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args, process_id)
        self.target_bpe = build_bpe(
            Namespace(
                **S2TDataConfig(
                    Path(self.args.fairseq_data) / self.args.fairseq_config
                ).bpe_tokenizer
            )
        )
        tgt_dict = self.model.decoder.dictionary
        self.blank = tgt_dict.bos()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.pad = tgt_dict.pad()
        self.blank_penalty = args.blankpen
        self.onlinefeatextractor = OnlineFeatureExtractor(args)
        assert self.init_target_index == self.blank
        self.step_size = self.model.encoder.step_size

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

        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state['model'], strict=False)
        self.model.eval()
        self.model.share_memory()
        self.model.to(self.device)

    @staticmethod
    def add_args(parser):
        FairseqSimulS2TAgent.add_args(parser)
        parser.add_argument(
            "--blankpen",
            type=float,
            default=1.0,
            help="Blank penalty to adjust src/tgt generation ratio.",
        )
        parser.add_argument(
            "--user-dir",
            type=str,
            default="examples/transformer_transducer",
        )
        parser.add_argument(
            "--sample-rate",
            type=int,
            default=16000,
            help="audio sampling rate",
        )
        parser.add_argument(
            "--feature-dim",
            type=int,
            default=80,
            help="number of fbank features",
        )
        parser.add_argument(
            "--frame-shift",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--frame-length",
            type=int,
            default=25,
        )
        parser.add_argument(
            "--global-cmvn",
            type=str,
            default="",
        )

    @property
    def min_input_length(self):
        conv = (
            self.model.encoder.subsample.conv
        )
        length = conv[-2].kernel_size[0]
        for layer in conv:
            if isinstance(layer, nn.Conv2d):
                length *= layer.stride[0]
        return length*self.args.frame_shift

    def reset(self):
        super().reset()
        self.states['read_inputs'] = 0
        self.states['tgt_encoder_state'] = None
        self.states['src_tokens'] = None
        if hasattr(self, "onlinefeatextractor"):
            self.onlinefeatextractor.clear_cache()

    def update_tgt_encoder(self):
        # predict tokens one by one
        curr_tgt_lengths = torch.LongTensor([1]).to(self.device)
        if self.target_length == 0:
            curr_tgt_tokens = torch.LongTensor([self.init_target_index]).to(self.device).unsqueeze(0)
        else:
            curr_tgt_tokens = torch.LongTensor([self.states['target_indices'][-1]]).to(self.device).unsqueeze(0)
        tgt_encoder_out = self.model.forward_tgt_encoder(
            tgt_tokens=curr_tgt_tokens,
            tgt_lengths=curr_tgt_lengths,
            incremental_state=self.states['incremental_states'],
        )
        self.states['tgt_encoder_state'] = self.model.get_tgt_encoder_outputs(tgt_encoder_out)[:1]

    def add_src_tokens(self, src_tokens):
        if self.states['src_tokens'] is None:
            self.states['src_tokens'] = src_tokens
        else:
            self.states['src_tokens'] = torch.cat((self.states['src_tokens'], src_tokens), dim=0)

    def pop_src_tokens(self, force=False):
        if self.states['src_tokens'] == None:
            return None, 0
        if force:
            ret_num = self.states['src_tokens'].size(0)
        else:
            ret_num = self.states['src_tokens'].size(0) // self.step_size * self.step_size

        if ret_num == 0:
            return None, ret_num
        ret = self.states['src_tokens'][:ret_num]
        self.states['src_tokens'] = None if ret_num == self.states['src_tokens'].size(0) else self.states['src_tokens'][ret_num:]
        return ret, ret_num


    def get_input_features(self, source_info):
        segment = np.array(source_info["segment"])
        segment *= 2**15
        self.is_finish_read = source_info["finished"]
        src_tokens = self.onlinefeatextractor(segment.tolist()).to(self.device)
        self.add_src_tokens(src_tokens)
        src_tokens, src_length = self.pop_src_tokens(force=self.is_finish_read)
        if src_length == 0:
            return None, 0

        return src_tokens.unsqueeze(0), torch.LongTensor([src_tokens.size(0)]).to(self.device)


    def process_read(self, source_info: Dict) -> Dict:
        self.states["encoder_states"] = None
        torch.cuda.empty_cache()
        src_tokens, src_lengths = self.get_input_features(source_info)
        if src_lengths > 0:
            self.states["encoder_states"] = self.model.forward_src_encoder(
                src_tokens, src_lengths, incremental_state=self.states['incremental_states']
            )
        return source_info

    @property
    def src_encoder_state(self):
        return self.states["encoder_states"]["encoder_out"][0] if self.states["encoder_states"] is not None else None

    def get_token_from_index(self, pred):

        self.states["target_indices"].append(pred)

        bpe_token = self.model.decoder.dictionary.string([pred])

        if re.match(r"^\[.+_.+\]$", bpe_token) or len(bpe_token) == 0:
            # Language Indicator or UNK
            return None

        return bpe_token

    def policy(self) -> None:
        # Read at the beginning
        while self.states["encoder_states"] is None:
            if self.is_finish_read:
                self.finish_eval()
                return
            self.read()
        if self.states['tgt_encoder_state'] is None:
            self.update_tgt_encoder()

        pred = self.blank
        if self.src_encoder_state is not None:

            while self.states['read_inputs'] < self.src_encoder_state.size(0):
                logits = self.model.get_logits(
                    enc_states=self.src_encoder_state[self.states['read_inputs'] : self.states['read_inputs'] + 1],
                    dec_states=self.states['tgt_encoder_state']
                )
                lprobs = torch.log_softmax(logits, dim=-1).view(-1)
                lprobs[self.pad] = -math.inf
                if self.target_length >= self.max_len:
                    lprobs[:] = -math.inf
                    lprobs[self.eos] = 0
                else:
                    lprobs[self.eos] = -math.inf
                    lprobs[self.unk] = -math.inf
                    lprobs[self.pad] = -math.inf
                if self.blank_penalty > 0:
                    lprobs[self.blank] -= self.blank_penalty

                pred = lprobs.argmax(dim=-1).item()

                if pred == self.blank:
                    if self.states['read_inputs'] == len(self.src_encoder_state) - 1:
                        if self.is_finish_read:
                            pred = self.eos
                        else:
                            self.states['read_inputs'] = 0
                    else:
                        self.states['read_inputs'] += 1
                        continue
                break

            # Make decision on model output

        if pred == self.blank and not self.is_finish_read:
            # Read
            self.read()
        else:
            # Predict
            token = self.get_token_from_index(pred)
            self.update_tgt_encoder()
            # Check if finish then write
            if not self.check_finish_eval():
                self.write(token)
