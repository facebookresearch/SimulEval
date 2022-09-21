import io
import sys
import logging
from argparse import Namespace

FAIRSEQ_SUPPORTED_SOURCE_MEDIUM = ["speech"]

logger = logging.getLogger("simuleval.utils.fairseq")

try:
    from fairseq.data.audio.audio_utils import (
        read_from_stored_zip,
        is_sf_audio_data,
        parse_path,
    )
except:
    pass  # Don't worry we will check somewhere else :)


def get_audio_file_path(path_of_fp):
    _path, slice_ptr = parse_path(path_of_fp)
    if len(slice_ptr) == 2:
        byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        assert is_sf_audio_data(byte_data)
        path_of_fp = io.BytesIO(byte_data)
    return path_of_fp


def use_fairseq(args: Namespace) -> bool:
    if args.fairseq_manifest is not None:
        return True
    if args.fairseq_data is not None:
        return True
    if args.fairseq_config is not None:
        return True
    return False


def check_fairseq_args(args: Namespace) -> None:
    try:
        import fairseq
    except:
        logger = logging.getLogger(
            "Please install Fairseq if you want to use Fairseq data."
        )
        sys.exit(1)

    if not (args.source is None and args.target is None):
        logger.error("Use either fairseq manifest or source/target, not both.")
        sys.exit(1)

    if args.source_type not in FAIRSEQ_SUPPORTED_SOURCE_MEDIUM:
        logger.error(f"Source type {args.source_type} is not suppored yet.")
        sys.exit(1)

    if args.fairseq_manifest:
        if args.fairseq_data or args.fairseq_config or args.fairseq_gen_subset:
            logger.error(
                "I already have a manifest, too many fairseq options were given."
            )
            sys.exit(1)
    else:
        if not (args.fairseq_data and args.fairseq_config and args.fairseq_gen_subset):
            logger.error(
                "I need fairseq data, config and gen_subset. Something is missing."
            )
            sys.exit(1)
