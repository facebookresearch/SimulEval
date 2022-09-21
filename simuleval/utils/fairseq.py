import sys
import logging
from argparse import Namespace

FAIRSEQ_SUPPORTED_SOURCE_MEDIUM = ["speech"]


def use_fairseq(args: Namespace) -> bool:
    if args.fairseq_manifest is not None:
        return True
    if args.fairseq_data is not None:
        return True
    if args.fairseq_config is not None:
        return True
    return False


def check_fairseq_args(args: Namespace) -> None:
    logger = logging.getLogger("simuleval.fairseq_args_check")
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
