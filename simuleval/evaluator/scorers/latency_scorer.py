from simuleval.metrics.latency import (
    AverageLagging,
    AverageProportion,
    DifferentiableAverageLagging,
)
from statistics import mean
from pathlib import Path
import subprocess
import logging
import textgrid
import sys
import shutil

logger = logging.getLogger("simuleval.latency_scorer")

LATENCY_SCORERS_DICT = {}


def register_latency_scorer(name):
    def register(cls):
        LATENCY_SCORERS_DICT[name] = cls
        return cls

    return register


class LatencyScorer:
    metric = None

    def __init__(self) -> None:
        pass

    def __call__(self, instances) -> float:
        raise NotImplementedError


def simul_trans_latency_scorer(metric):
    def create_class(klass):
        class Klass(klass):
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
                    scores.append(metric(delays, src_len, tgt_len).item())

                return mean(scores)

        return Klass

    return create_class


@register_latency_scorer("AL")
@simul_trans_latency_scorer(AverageLagging)
class ALScorer(LatencyScorer):
    """Average Lagging scorers

    Usage:
        --latency-metrics AL
    """
    pass


@register_latency_scorer("AP")
@simul_trans_latency_scorer(AverageProportion)
class APScorer(LatencyScorer):
    """Average Proportion scorers

    Usage:
        --latency-metrics AP
    """
    pass


@register_latency_scorer("DAL")
@simul_trans_latency_scorer(DifferentiableAverageLagging)
class DALScorer(LatencyScorer):
    """Differentiable Average Lagging scorers

    Usage:
        --latency-metrics DAL
    """
    pass


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
