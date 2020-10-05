# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . states import BaseStates


class SpeechStates(BaseStates):
    def init_entries(self):
        super().init_entries()

    def get_info_from_server(self, num_segments):
        info = self.client.get_source(
            self.instance_id,
            {"segment_size": num_segments * self.agent.speech_segment_size}
        )

        self.sample_rate = info["sample_rate"]
        return info

    def num_samples(self):
        return sum(len(x) for x in self.segments.source)

    def num_milliseconds(self):
        sample_rate = getattr(self, "sample_rate", 0)

        if sample_rate != 0:
            return round(sum(len(x) * 1000 / sample_rate for x in self.segments.source))

        return 0

    def summarize(self):
        return {
            "instance_id": self.instance_id,
            "finish_read": self.finish_read(),
            "finish_hypo": self.finish_hypo(),
            "segments": {
                "source": {
                    "ms": self.num_milliseconds(),
                    "num_samples": self.num_samples(),
                    "num_samples_queue": sum(len(x) for x in self.unit_queue.source.value),
                },
                "target": self.segments.target.info(),
            },
            "units": {
                "source": self.units.source.info(),
                "target": self.units.target.info(),
            },
            "unit_queue": {
                "target": self.unit_queue.target.info(),
            },
        }

    @property
    def source(self):
        return self.units.source

    @property
    def target(self):
        return self.units.target
