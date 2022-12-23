from statistics import mean
from pathlib import Path
import subprocess
import logging
import textgrid
import sys
import shutil
from typing import List, Union
from simuleval.evaluator.instance import TextInputInstance
from simuleval.evaluator.instance import INSTANCE_TYPE_DICT

logger = logging.getLogger("simuleval.latency_scorer")

LATENCY_SCORERS_DICT = {}


def register_latency_scorer(name):
    def register(cls):
        LATENCY_SCORERS_DICT[name] = cls
        return cls

    return register


class LatencyScorer:
    metric = None

    def __init__(
        self, computation_aware: bool = False, use_ref_len: bool = True
    ) -> None:
        super().__init__()
        self.use_ref_len = use_ref_len
        self.computation_aware = computation_aware

    @property
    def timestamp_type(self):
        return "delays" if not self.computation_aware else "elapsed"

    def __call__(self, instances) -> float:
        scores = []
        for index, ins in instances.items():
            if isinstance(ins, TextInputInstance):
                if self.computation_aware:
                    raise RuntimeError(
                        "The computation aware latency is not supported on text input."
                    )
            delays = getattr(ins, self.timestamp_type, None)
            if delays is None or len(delays) == 0:
                logger.warn(f"{index} instance has no delay information. Skipped")
                continue

            if not self.use_ref_len or ins.reference is None:
                tgt_len = len(delays)
            else:
                tgt_len = len(ins.reference.split())
            src_len = ins.source_length
            scores.append(self.compute(delays, src_len, tgt_len))

        return mean(scores)


@register_latency_scorer("AL")
class ALScorer(LatencyScorer):
    r"""
    Average Lagging (AL) from
    `STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework <https://arxiv.org/abs/1810.08398>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        AL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{|Y|}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length

    Usage:
        ----latency-metrics AL
    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ):
        """
        Function to compute latency on one sentence (instance).

        Args:
            delays (List[Union[float, int]]): Sequence of delays.
            source_length (Union[float, int]): Length of source sequence.
            target_length (Union[float, int]): Length of target sequence.

        Returns:
            float: the latency score on one sentence.
        """

        if delays[0] > source_length:
            return delays[0]

        AL = 0
        gamma = target_length / source_length
        tau = 0
        for t_miuns_1, d in enumerate(delays):
            if d <= source_length:
                AL += d - t_miuns_1 / gamma
                tau = t_miuns_1 + 1

                if d == source_length:
                    break
        AL /= tau
        return AL


@register_latency_scorer("LAAL")
class LAALScorer(ALScorer):
    r"""
    Length Adaptive Average Lagging (LAAL) as proposed in
    `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
    <https://arxiv.org/abs/2204.06028>`_.
    The name was suggested in `Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation
    <https://arxiv.org/abs/2206.05807>`_.
    It is the original Average Lagging as proposed in
    `Controllable Latency using Prefix-to-Prefix Framework
    <https://arxiv.org/abs/1810.08398>`_
    but is robust to the length differece between the hypothesis and reference.

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        LAAL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{max(|Y|,|Y*|)}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length, and :math:`|Y*|` is the length of the hypothesis.

    Usage:
        ----latency-metrics LAAL
    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ):
        """
        Function to compute latency on one sentence (instance).

        Args:
            delays (List[Union[float, int]]): Sequence of delays.
            source_length (Union[float, int]): Length of source sequence.
            target_length (Union[float, int]): Length of target sequence.

        Returns:
            float: the latency score on one sentence.
        """

        if delays[0] > source_length:
            return delays[0]

        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_miuns_1, d in enumerate(delays):
            if d <= source_length:
                LAAL += d - t_miuns_1 / gamma
                tau = t_miuns_1 + 1

                if d == source_length:
                    break
        LAAL /= tau
        return LAAL


@register_latency_scorer("AP")
class APScorer(LatencyScorer):
    r"""
    Average Proportion (AP) from
    `Can neural machine translation do simultaneous translation? <https://arxiv.org/abs/1606.02012>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,
    the AP is calculated as:

    .. math::

        AP = \frac{1}{|X||Y]} \sum_i^{|Y|} D_i

    Usage:
        ----latency-metrics AP
    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ) -> float:
        """
        Function to compute latency on one sentence (instance).

        Args:
            delays (List[Union[float, int]]): Sequence of delays.
            source_length (Union[float, int]): Length of source sequence.
            target_length (Union[float, int]): Length of target sequence.

        Returns:
            float: the latency score on one sentence.
        """
        return sum(delays) / (source_length * target_length)


@register_latency_scorer("DAL")
class DALScorer(LatencyScorer):
    r"""
    Differentiable Average Lagging (DAL) from
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/abs/1906.05218)

    Usage:
        ----latency-metrics DAL
    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ):
        """
        Function to compute latency on one sentence (instance).

        Args:
            delays (List[Union[float, int]]): Sequence of delays.
            source_length (Union[float, int]): Length of source sequence.
            target_length (Union[float, int]): Length of target sequence.

        Returns:
            float: the latency score on one sentence.
        """
        DAL = 0
        target_length = len(delays)
        gamma = target_length / source_length
        g_prime_last = 0
        for i_miuns_1, g in enumerate(delays):
            if i_miuns_1 + 1 == 1:
                g_prime = g
            else:
                g_prime = max([g, g_prime_last + 1 / gamma])

            DAL += g_prime - i_miuns_1 / gamma
            g_prime_last = g_prime

        DAL /= target_length
        return DAL


@register_latency_scorer("ATD")
class ATDScorer(LatencyScorer):
    r"""
    Average Token Delay (ATD) from
    Average Token Delay: A Latency Metric for Simultaneous Translation
    (https://arxiv.org/abs/2211.13173)

    Different from speech segments, text tokens have no length
    and multiple tokens can be output at the same time like subtitle.
    Therefore, we set its length to be 0. However, to calculate latency in text-text,
    we give virtual time 1 for the length of text tokens.

    Usage:
        ----latency-metrics ATD
    """

    def __call__(self, instances) -> float:
        if isinstance(instances[0], INSTANCE_TYPE_DICT["text-text"]):
            TGT_TOKEN_LEN = 1
            SRC_TOKEN_LEN = 1
            INSTANCE_TYPE = "text-text"
        elif isinstance(instances[0], INSTANCE_TYPE_DICT["speech-text"]):
            TGT_TOKEN_LEN = 0
            SRC_TOKEN_LEN = 300  #300ms per word
            INSTANCE_TYPE = "speech-text"
        elif  isinstance(instances[0], INSTANCE_TYPE_DICT["speech-speech"]):
            TGT_TOKEN_LEN = 300
            SRC_TOKEN_LEN = 300
            INSTANCE_TYPE = "speech-speech"
        else:
            logger.warn(f"{index} instance type is not expected. Skipped")
            return

        scores = []
        for index, ins in instances.items():
            delays = getattr(ins, "delays", None)
            if delays is None:
                logger.warn(f"{index} instance has no delay information. Skipped")
                continue

            if self.computation_aware:
                elapsed = getattr(ins, "elapsed", None)
                if elapsed is None:
                    logger.warn(f"{index} instance has no computational delay information. Skipped")
                    continue
            else:
                elapsed = [0] * len(delays)

            chunk_lens = {"src":[0],"tgt":[0]}
            token_to_chunk = {"src":[0],"tgt":[0]}
            token_to_time = {"src":[0],"tgt":[0]}
            tgt_token_lens = []

            if INSTANCE_TYPE == "speech-speech":
                s2s_delays = []
                s2s_elapsed = []
                chunk_lens["tgt"] += ins.duration
                for i, chunk_len in enumerate(chunk_lens,1):
                    num_tokens, rest = divmod(chunk_len, TGT_TOKEN_LEN)
                    token_lens = num_tokens * [TGT_TOKEN_LEN] + [rest]
                    s2s_delays += delays[i-1] * len(token_lens)
                    s2s_elapsed += elapsed[i-1] * len(token_lens)
                    token_to_chunk["tgt"] += [i] * len(token_lens)
                    tgt_token_lens += token_lens
                delays = s2s_delays
                elapsed = s2s_elapsed
            else:
                prev_delay = None
                for delay in delays:
                    if delay != prev_delay:
                        chunk_lens["tgt"].append(1)
                    else:
                        chunk_lens["tgt"][-1] += 1
                    prev_delay = delay
                for i, chunk_len in enumerate(chunk_lens["tgt"][1:],1):
                    token_to_chunk["tgt"] +=  [ i ] * chunk_len
                tgt_token_lens = [TGT_TOKEN_LEN] * len(delays)

            if self.computation_aware and elapsed != [0] * len(delays):
                compute_elapsed = self.subtract(elapsed, delays)
                compute_times = self.subtract(compute_elapsed, [0] + compute_elapsed[:-1])
            else:
                compute_times = elapsed

            delays_no_duplicate = sorted(set(delays), key=delays.index)
            chunk_lens["src"] += self.subtract(delays_no_duplicate, [0] + delays_no_duplicate[:-1])

            for i, chunk_len in enumerate(chunk_lens["src"][1:],1):
                if INSTANCE_TYPE == "text-text":
                    token_lens = chunk_len * [SRC_TOKEN_LEN]
                else:
                    num_tokens, rest = divmod(chunk_len, SRC_TOKEN_LEN)
                    token_lens = num_tokens * [SRC_TOKEN_LEN] + [rest]
                for token_len in token_lens:
                    token_to_time["src"].append(token_to_time["src"][-1] + token_len)
                    token_to_chunk["src"].append(i)

            for delay, compute_time, token_len in zip(delays, compute_times, tgt_token_lens):
                tgt_start_time = max(delay, token_to_time["tgt"][-1])
                token_to_time["tgt"].append(tgt_start_time +token_len + compute_time)

            scores.append(self.compute(chunk_lens, token_to_chunk, token_to_time))

        return mean(scores)

    def subtract(self, arr1, arr2):
        return [x - y for x, y in zip(arr1, arr2)]

    def compute(
        self,
        chunk_lens: dict,
        token_to_chunk: dict,
        token_to_time: dict,
    ) -> float:
        """
        Function to compute latency on one sentence (instance).
        Args:
            delays (List[Union[float, int]]): Sequence of delays.
        Returns:
            float: the latency score on one sentence.
        """

        tgt_to_src = []

        for t in range(1, len(token_to_chunk["tgt"])):
            chunk_id = token_to_chunk["tgt"][t]
            AccLen_x = sum(chunk_lens["src"][:chunk_id])
            AccLen_y = sum( chunk_lens["tgt"][:chunk_id])

            S = t - max(0, AccLen_y - AccLen_x)
            current_src_len = sum(chunk_lens["src"][:chunk_id + 1])
            if S < current_src_len:
                tgt_to_src.append((t, S))
            else:
                tgt_to_src.append((t, current_src_len))

        atd_delays = []
        for t, s  in tgt_to_src:
            atd_delay = token_to_time["tgt"][t] - token_to_time["src"][s]
            atd_delays.append(atd_delay)

        return float(mean(atd_delays))


@register_latency_scorer("StartOffset")
class StartOffsetScorer(LatencyScorer):
    """Starting offset of the translation

    Usage:
        ----latency-metrics StartOffset

    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ):
        return delays[0]


@register_latency_scorer("EndOffset")
class EndOffsetScorer(LatencyScorer):
    """Ending offset of the translation

    Usage:
        ----latency-metrics EndOffset

    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ):
        return delays[-1] - source_length


@register_latency_scorer("RTF")
class RTFScorer(LatencyScorer):
    """Compute Real Time Factor (RTF)

    Usage:
        ----latency-metrics (RTF)

    """

    def compute(
        self,
        delays: List[Union[float, int]],
        source_length: Union[float, int],
        target_length: Union[float, int],
    ):
        return delays[-1] / source_length


def speechoutput_aligment_latency_scorer(scorer_class):
    class Klass(scorer_class):
        def __init__(self) -> None:
            assert getattr(self, "boundary_type", None) in [
                "BOW",
                "EOW",
                "COW",
            ], self.boundary_type
            super().__init__()
            if self.computation_aware:
                raise RuntimeError(
                    "The computation aware latency for speech output is not supported yet"
                )

        @property
        def timestamp_type(self):
            return "aligned_delays"

        def __call__(self, instances) -> float:
            self.prepare_alignment(instances)
            return super().__call__(instances)

        def prepare_alignment(self, instances):
            try:
                subprocess.check_output("mfa -h", shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as grepexc:
                logger.error(grepexc.output.decode("utf-8").strip())
                logger.error("Please make sure the mfa is correctly installed.")
                sys.exit(1)

            output_dir = Path(instances[0].prediction).absolute().parent.parent
            align_dir = output_dir / "align"
            if not align_dir.exists():
                logger.info("Align target transcripts with speech.")
                temp_dir = Path(output_dir) / "mfa"
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir.mkdir(exist_ok=True)
                original_model_path = Path.home() / "Documents/MFA/pretrained_models"
                acoustic_model_path = temp_dir / "acoustic.zip"
                acoustic_model_path.symlink_to(
                    original_model_path / "acoustic" / "english_mfa.zip"
                )
                dictionary_path = temp_dir / "dict"
                dictionary_path.symlink_to(
                    original_model_path / "dictionary" / "english_mfa.dict"
                )
                mfa_command = f"mfa align {output_dir  / 'wavs'} {dictionary_path.as_posix()} {acoustic_model_path.as_posix()} {align_dir.as_posix()} --clean --overwrite --temporary_directory  {temp_dir.as_posix()}"
                logger.info(mfa_command)

                subprocess.run(
                    mfa_command,
                    shell=True,
                    check=True,
                )
            else:
                logger.info("Found existing alignment")

            for file in align_dir.iterdir():
                if file.name.endswith("TextGrid"):
                    index = int(file.name.split("_")[0])
                    target_offset = instances[index].delays[0]
                    info = textgrid.TextGrid.fromFile(file)
                    delays = []
                    for interval in info[0]:
                        if len(interval.mark) > 0:
                            if self.boundary_type == "BOW":
                                delays.append(target_offset + 1000 * interval.minTime)
                            elif self.boundary_type == "EOW":
                                delays.append(target_offset + 1000 * interval.maxTime)
                            else:
                                delays.append(
                                    target_offset
                                    + 0.5 * (interval.maxTime + interval.minTime) * 1000
                                )
                    setattr(instances[index], self.timestamp_type, delays)

    return Klass


for boundary_type in ["BOW", "COW", "EOW"]:
    for metric in ["AL", "LAAL", "AP", "DAL", "ATD", "StartOffset", "EndOffset"]:

        @register_latency_scorer(f"{metric}_SpeechAlign_{boundary_type}")
        @speechoutput_aligment_latency_scorer
        class SpeechAlignScorer(LATENCY_SCORERS_DICT[metric]):
            f"""Compute {metric} based on alignment ({boundary_type})

            Usage:
                ----latency-metrics {metric}_SpeechAlign_{boundary_type}
            """
            boundary_type = boundary_type
            __name__ = f"{metric}SpeechAlign{boundary_type}Scorer"
