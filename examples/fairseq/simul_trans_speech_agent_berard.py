# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . simul_trans_speech_agent import SimulTransSpeechAgent
from . import register_agent


@register_agent("speech_berard")
class SimulTransSpeechAgentBerard(SimulTransSpeechAgent):
    def segment_size_from_states(self, states):
        return self.model.decoder.attention.segment_size(
            self.frame_shift,
            self.model.subsampling_factor()
        )

    def decision_from_states(self, states):
        self.apply_cmvn(states)
        return self.model.decision_from_states(states)

    def predict_from_states(self, states):
        self.apply_cmvn(states)
        return self.model.predict_from_states(states)

    @staticmethod
    def apply_cmvn(states):
        from examples.simultaneous_translation.data.data_utils import apply_mv_norm
        if len(states["indices"]["src"]) > 0:
            states["indices"]["src"] = apply_mv_norm(states["indices"]["src"])
            states["steps"]["src"] = states["speech_steps"]['milisecond']
