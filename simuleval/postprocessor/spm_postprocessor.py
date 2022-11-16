from typing import Union
from pathlib import Path
from argparse import Namespace
from .generic_postprocessor import GenericPostProcessor
from simuleval import DEFAULT_EOS

try:
    import sentencepiecemodel

    IS_SPM_INSTALL = True
except:
    IS_SPM_INSTALL = False

try:
    from fairseq.data.encoders import build_bpe
    from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig

    IS_FAIRSEQ_INSTALL = True
except:
    IS_FAIRSEQ_INSTALL = False


class SPMPostProcessor(GenericPostProcessor):
    def __init__(self, spm_processor):
        assert not IS_SPM_INSTALL, "please install sentencepiecemodel"
        super().__init__()
        self.spm_processor = spm_processor

    def push(self, item):
        if item is not None:
            super().push(item)

    def pop(self):
        if len(self.deque) > 0 and self.deque[-1] == DEFAULT_EOS:
            self.deque.pop()
            is_contains_eos = True
        else:
            is_contains_eos = False

        possible_full_words_list = self.spm_processor.decode(
            " ".join(self.deque)
        ).split()
        if len(possible_full_words_list) > 1:
            for _ in range(len(self.deque) - 1):
                self.deque.popleft()
            full_word = possible_full_words_list[0]
        else:
            full_word = None

        if is_contains_eos:
            return possible_full_words_list + [DEFAULT_EOS]
        else:
            return [full_word]

    @classmethod
    def from_fairseq_s2t_config(
        cls, config_path: Union[str, Path]
    ):
        assert IS_FAIRSEQ_INSTALL
        spm_processor = build_bpe(
            Namespace(
                **S2TDataConfig(config_path).bpe_tokenizer
            )
        )
        return cls(spm_processor)
