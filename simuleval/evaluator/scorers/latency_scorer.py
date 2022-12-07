from statistics import mean
from pathlib import Path
import subprocess
import logging
import textgrid
import sys
import shutil
from typing import List, Union

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
        for ins in instances.values():
            delays = getattr(ins, self.timestamp_type)
            if self.use_ref_len or ins.reference is None:
                tgt_len = ins.prediction_length
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
        AL = 0
        target_length
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


def speechoutput_aligment_latency_scorer(scorer_class):
    class Klass(scorer_class):
        def __init__(self) -> None:
            assert getattr(self, "boundary_type", None) in [
                "BOW",
                "EOW",
                "COW",
            ], self.boundary_type
            super().__init__()

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
            if True:  # not align_dir.exists():
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

                import pdb

                pdb.set_trace()
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
    for metric in ["AL", "AP", "DAL"]:

        @register_latency_scorer(f"{metric}_SpeechAlign_{boundary_type}")
        @speechoutput_aligment_latency_scorer
        class SpeechAlignScorer(LATENCY_SCORERS_DICT[metric]):
            boundary_type = boundary_type
            __name__ = f"{metric}SpeechAlign{boundary_type}Scorer"
