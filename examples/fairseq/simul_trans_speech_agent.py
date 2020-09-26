# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from . import register_agent
from simuleval.states import SpeechStates
from simuleval.agents import Agent

import torch

READ = 0
WRITE = 1


@register_agent("speech")
class SpeechAgent(Agent):
    def init_states(self):
        return SpeechStates(self.args, self.model)

    def read_action(self, states):
        segment_size = self.segment_size_from_states(states)
        return {'key': "get", 'value': {"segment_size": segment_size}}

    def build_word_splitter(self, args):
        self.word_splitter = {}

        self.word_splitter["tgt"] = eval(f"{args.tgt_splitter_type}WordSplitter")(
            getattr(args, "tgt_splitter_path"))

    @staticmethod
    def _add_pad(list_tensor, list_length, batch_first=True):
        max_len = max(list_length)
        tensor = torch.zeros(len(list_tensor), max_len,
                             list_tensor[0].size(1)).type_as(list_tensor[0])
        for i, t in enumerate(list_tensor):
            tensor[i, :list_length[i]] = t
        return tensor

    def segment_size_from_states(self, states):
        return self.model.decoder.layers[0].encoder_attn.pooling_ratio * \
            4 * self.frame_shift

    @torch.no_grad()
    def predict_from_states(self, states):

        decoder_states = self.model.decoder.output_layer(
            states["decoder_features"]
        )
        log_probs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]],
            log_probs=True
        )

        index = log_probs.argmax(dim=-1)

        token = self.model.decoder.dictionary.string(index)

        torch.cuda.empty_cache()

        return token, index[0, 0].item()

    @torch.no_grad()
    def decision_from_states(self, states):
        '''
        This funcion take states dictionary as input, and gives the agent
        a decision of whether read a token from server. Moreover, the decoder
        states are also calculated here so we can directly generate a target
        token without recompute every thing
        '''

        if "encoder_states" not in states:
            if states["finish_read"]:
                import pdb
                pdb.set_trace()
            return READ

        # online means we still need tokens to feed the model
        states["model_states"]["online"] = not (
            states["finish_read"]
        )

        # TODO clumsy here, refactor later
        states["model_states"]["steps"] = {
            "src": states["steps"]["src"],
            "tgt": 1 + len(states["tokens"]["tgt"])
        }

        tgt_indices = self.to_device(torch.LongTensor(
            [
                [self.model.decoder.dictionary.eos()]
                + states["indices"]["tgt"]
            ]
        ))

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states["encoder_states"],
            incremental_state=states["model_states"],
            features_only=True,
        )
        torch.cuda.empty_cache()

        states["decoder_features"] = x

        return outputs["action"]
