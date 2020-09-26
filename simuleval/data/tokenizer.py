# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DUMMY_SENTENCE = "AA BB CC"


class Tokenizer(object):
    def __init__(self):
        self._check_implementation(DUMMY_SENTENCE)

    def _check_implementation(self, string: str):
        process_str = self.process_line(string)
        tokens = self.split(process_str)
        assert isinstance(tokens, list)
        eow_indices = self.eow_indices(tokens)
        assert isinstance(eow_indices, list)
        assert len(eow_indices) > 0
        for eos_idx in eow_indices:
            self.merge(tokens[:eos_idx + 1])
            tokens = tokens[eos_idx + 1:]

    def preprocess(self, string: str):
        return string

    def eow_indices(self, iterable):
        # return a list of end of word indices
        # return empty list if there is none
        raise NotImplementedError

    def split(self, string: str):
        raise NotImplementedError

    def merge(self, list_of_tokens):
        raise NotImplementedError

    def postprocess(self, string: str):
        return string


class NoneTokenizer(object):
    def __init__(self, model):
        pass

    def split(self, string):
        return [string]

    def process_line(self, string):
        return [string]

    def finished_word(self, string):
        return True

    def merge(self, list_of_string):
        return "".join(list_of_string)

    def last_full_word_step(self, tokens, step):
        return len(tokens)

    def end_idx_last_full_word(self, tokens):
        return len(tokens)


class BPEWordSplitter(object):
    # TODO: lock back here
    def __init__(self, model_path):
        super().__init__()
        from subword_nmt.apply_bpe import BPE
        with open(model_path) as f:
            self.model = BPE(f)

    def split(self, string):
        return self.model.process_line(string).split()

    def end_idx_last_full_word(self, tokens):
        # Begin of word indices
        bow_indices = [0] + [i + 1 for i,
                             t in enumerate(tokens[1:]) if t[-2:] != '@@']

        if len(bow_indices) < 2:
            return 0
        else:
            return bow_indices[-1]

    def merge(self, list_of_string):
        return " ".join([item.replace("@@", "") for item in list_of_string])


class SentencePieceModelTokenizer(Tokenizer):
    def __init__(self, model_path):
        super().__init__()
        import sentencepiece as spm
        self.model = spm.SentencePieceProcessor()
        self.model.Load(model_path)

    def split(self, string):
        return self.model.EncodeAsPieces(string)

    def end_idx_last_full_word(self, tokens):
        # Begin of word indices
        bow_indices = [i for i, t in enumerate(tokens) if t[0] == '\u2581']

        if len(bow_indices) < 2:
            return 0
        else:
            return bow_indices[-1]

    def merge(self, list_of_string):
        return self.model.DecodePieces(list_of_string)

    def extract_full_word(self):
        pass


# SPLITTER_DICT = {
#    None: NoneWordSplitter,
#    "BPE": BPEWordSplitter,
#    "SentencePieceModel": SentencePieceModelWordSplitter,
# }
