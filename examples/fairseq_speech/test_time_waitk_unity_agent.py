import os
import sys
import json
import torch
import simuleval
from argparse import Namespace, ArgumentParser
from typing import Optional, List

sys.path.append(os.path.join(simuleval.__path__[0], "..", "examples"))
from fairseq_speech.generic_agent import FairseqSimulS2SAgent
from fairseq_speech.utils import test_time_waitk_agent
from simuleval.postprocessor import GenericPostProcessor
from simuleval import DEFAULT_EOS
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder


class CodeHiFiGANVocoderPostProcessor(GenericPostProcessor):
    def __init__(self, args):
        super().__init__()
        self.fs = 16000
        self.args = args
        self.dur_prediction = args.dur_prediction
        self.device = torch.device(args.device[0])
        with open(args.vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
            self.vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)
            self.vocoder.to(self.device)

    def push(self, unit_list_str: str):
        if unit_list_str == DEFAULT_EOS:
            return
        self.unit_buffer = [int(x) for x in unit_list_str.split()]

    def pop(self):
        if len(self.unit_buffer) == 0:
            return None
        x = {
            "code": torch.LongTensor(self.unit_buffer).view(1, -1).to(self.device),
        }
        wav = self.vocoder(x, self.dur_prediction).cpu().tolist()
        self.unit_buffer = []
        return json.dumps({"samples": wav, "sample_rate": self.fs}).replace(" ", "")

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--vocoder", type=str, required=True, help="path to the CodeHiFiGAN vocoder"
        )
        parser.add_argument(
            "--vocoder-cfg",
            type=str,
            required=True,
            help="path to the CodeHiFiGAN vocoder config",
        )
        parser.add_argument(
            "--dur-prediction",
            action="store_true",
            help="enable duration prediction (for reduced/unique code sequences)",
        )
        parser.add_argument(
            "--max-len-a-mt",
            type=float,
            default=0,
            help="Max length of predictions, a in ax + b",
        )
        parser.add_argument(
            "--max-len-b-mt",
            type=float,
            default=200,
            help="Max length of predictions, b in ax + b",
        )


@test_time_waitk_agent
class FairseqTestWaitKUnitYAgent(FairseqSimulS2SAgent):
    def __init__(self, args: Namespace, process_id: Optional[int] = None) -> None:
        super().__init__(args, process_id)
        self.max_len_a_mt = args.max_len_a_mt
        self.max_len_b_mt = args.max_len_b_mt
        self.subword_bpe = []
        self.subword_offset = 0
        self.duration = 0
        with open(os.path.join(self.args.output, "debug.log"), "w") as f:
            pass

    def build_postprocessor(self, args: Namespace):
        return CodeHiFiGANVocoderPostProcessor(args)

    def reset(self):
        super().reset()
        self.states["target_mt_indices"] = []
        self.states["target_mt_decoder_features"] = []
        self.states["mt_incremental_states"] = {}
        self.is_finish_s2t = False
        self.is_mt_early_stop = False

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--target-letter-decoder-args",
            type=str,
            help="Extra target_letter decoder arguments",
            default=1,
        )
        parser.add_argument(
            "--vocoder-path",
            type=str,
            help="Path to vocoder",
        )
        CodeHiFiGANVocoderPostProcessor.add_args(parser)

    def get_mt_index_tensor(
        self,
    ) -> torch.LongTensor:
        # assert task in ["mt", "t2u"]
        eos_token_index = self.fairseq_task.tgt_dict_mt.index(
            self.fairseq_task.eos_token_mt
        )
        return (
            torch.LongTensor([eos_token_index] + self.states["target_mt_indices"])
            .to(self.device)
            .unsqueeze(0)
        )

    def get_t2u_encoder_out(self):
        torch.cuda.empty_cache()
        x = torch.cat(self.states["target_mt_decoder_features"], dim=0)
        if getattr(self.model, "proj", None) is not None:
            x = self.model.proj(x)

        if getattr(self.model, "synthesizer_encoder", None) is not None:
            t2u_encoder_out = self.model.synthesizer_encoder(x, None)
        else:
            t2u_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [],
                "encoder_embedding": [],
                "encoder_states": [],
                "src_tokens": [],
                "src_lengths": [],
            }
        if getattr(self.model, "t2u_augmented_cross_attn", False):
            encoder_outs_aug = [t2u_encoder_out]
        else:
            encoder_outs = [t2u_encoder_out]
            encoder_outs_aug = None
        return encoder_outs, encoder_outs_aug

    @property
    def max_len_mt(self):
        return self.max_len_a_mt * self.source_length + self.max_len_b_mt

    def is_mt_policy_read(self) -> bool:
        tgt_mt_indices = self.get_mt_index_tensor()
        self.states["mt_incremental_states"]["steps"] = self.get_current_steps(
            self.states["encoder_states"]["encoder_out"][0], tgt_mt_indices
        )
        self.states["mt_incremental_states"]["online"] = {
            "only": torch.tensor(not self.is_finish_read)
        }

        # Run wait-k policy on MT decoder
        mt_decoder = getattr(self.model, f"{self.model.mt_task_name}_decoder")
        mt_decoder_features, outputs = mt_decoder.forward(
            prev_output_tokens=tgt_mt_indices,
            encoder_out=self.states["encoder_states"],
            incremental_state=self.states["mt_incremental_states"],
            features_only=True,
        )
        mt_decoder_states = mt_decoder.output_layer(mt_decoder_features)

        # Make decision on model output
        if outputs.action == 0 and not self.is_finish_read:
            #  Read
            return True
        else:
            #  Predict
            log_probs = mt_decoder.get_normalized_probs(
                [mt_decoder_states[:, -1:]], log_probs=True
            )
            index = log_probs.argmax(dim=-1)[0, 0].item()
            self.states["target_mt_indices"].append(index)
            self.states["target_mt_decoder_features"].append(mt_decoder_features)

            if index == self.fairseq_task.tgt_dict_mt.eos():
                if not self.is_finish_read:
                    self.is_mt_early_stop = True

                    return False
                else:
                    self.is_finish_s2t = True

            self.subword_bpe.append(
                {
                    "token": self.fairseq_task.tgt_dict_mt.string([index]),
                    "time": len(self.states["source_samples"]) / 16000
                    + self.subword_offset,
                }
            )
            if len(self.states["target_mt_indices"]) > self.max_len_mt:
                self.is_finish_s2t = True
            return False

    def finish_eval(self) -> None:
        import codecs

        with codecs.open(
            os.path.join(self.args.output, "debug.log"), "a", "utf-8"
        ) as f:
            info = {
                "s2t": self.subword_bpe,
            }
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        self.subword_bpe = []
        self.subword_offset = 0

        return super().finish_eval()

    def reset_early_stop(self):
        keep_index = self.index
        self.subword_offset = (
            self.subword_bpe[-1]["time"] + self.source_segment_size / 1000
        )
        self.reset()
        self.index = keep_index

    def policy(self) -> None:
        # 0.0 Read at the beginning
        while self.states["encoder_states"] is None:
            if self.is_finish_read:
                self.finish_eval()
                return
            self.read()
        if self.is_mt_policy_read():
            # 1.2.1 Read
            self.read()
        else:
            # 1.2.2 After predict a text token
            encoder_outs, encoder_outs_aug = self.get_t2u_encoder_out()
            is_write = True
            pred_indices = []
            while is_write:
                # Keep predicting Unit
                tgt_indices = self.get_target_index_tensor()
                self.states["incremental_states"]["steps"] = self.get_current_steps(
                    torch.LongTensor(self.states["target_mt_indices"]), tgt_indices
                )
                self.states["incremental_states"]["online"] = {
                    "only": torch.tensor(
                        not (self.is_finish_read and self.is_finish_s2t)
                        and not self.is_mt_early_stop
                    )
                }
                decoder_state, outputs = self.model.decoder.forward(
                    tgt_indices,
                    encoder_outs[0],
                    self.states["incremental_states"],
                )
                is_write = outputs.action
                if not is_write:
                    break

                log_probs = self.model.decoder.get_normalized_probs(
                    [decoder_state[:, -1:]], log_probs=True
                )
                index = log_probs.argmax(dim=-1)[0, 0].item()

                if (
                    index == self.model.decoder.dictionary.eos()
                    or len(self.states["target_indices"]) > self.max_len
                ):
                    if self.is_finish_s2t:
                        self.finish_eval()
                    break

                self.states["target_indices"].append(index)
                pred_indices.append(index)
            if self.is_mt_early_stop:
                self.reset_early_stop()

            if len(pred_indices) > 0:
                self.write(self.model.decoder.dictionary.string(pred_indices))
