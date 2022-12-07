from typing import Union
from pathlib import Path
from argparse import Namespace
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents import TextToTextAgent
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


class SentencePieceModelDetokenizer(TextToTextAgent):
    def __init__(self, args, spm_processor):
        assert not IS_SPM_INSTALL, "please install sentencepiecemodel"
        super().__init__(args)
        self.spm_processor = spm_processor
        self.is_finished = False

    def reset(self):
        super().reset()
        self.is_finished = False

    def push(self, item: str) -> None:
        if item is None:
            return
        if item == DEFAULT_EOS:
            self.is_finished = True

        super().push(item)

    def pop(self):
        item = super().pop()
        if item is None:
            item = []
        return item

    def policy(self):
        if len(self.queue) > 0 and self.queue[-1] == DEFAULT_EOS:
            self.queue.pop()
            is_contains_eos = True
        else:
            is_contains_eos = False

        possible_full_words_list = self.spm_processor.decode(
            " ".join([x for x in self.queue])
        ).split()

        if len(possible_full_words_list) > 1 or is_contains_eos:
            for _ in range(len(self.queue) - 1):
                self.queue.popleft()
            full_word = possible_full_words_list[0]
            output_list = [full_word]
            if is_contains_eos:
                output_list += [DEFAULT_EOS]
            return WriteAction(output_list)

        else:
            return ReadAction()

    @classmethod
    def from_fairseq_s2t_config(cls, config_path: Union[str, Path], args):
        assert IS_FAIRSEQ_INSTALL
        spm_processor = build_bpe(Namespace(**S2TDataConfig(config_path).bpe_tokenizer))
        return cls(args, spm_processor)

    @classmethod
    def from_args(cls, args):
        return cls.from_fairseq_s2t_config(
            Path(args.fairseq_data) / args.fairseq_config, args
        )
