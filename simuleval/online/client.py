# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import requests
import logging
from typing import Optional


logger = logging.getLogger('simuleval.online.client')


class Client(object):

    def __init__(self, args):
        self.hostname = args.hostname
        self.port = args.port
        self.timeout = getattr(args, 'timeout', 10)
        self.args = args
        self.base_url = f'http://{self.hostname}:{self.port}'

    def reset_scorer(self):
        # start eval session

        url = f'{self.base_url}'

        try:
            _ = requests.post(url, timeout=self.timeout)
        except Exception as e:
            raise SystemExit(e)

    def get_scores(self, instance_id=None):
        # end eval session
        url = f'{self.base_url}/result'
        params = {"instance_id": instance_id}
        try:
            r = requests.get(url, params=params)
            return r.json()
        except Exception as e:
            logger.error(f'Failed to retreive scores: {e}')
            return None

    def get_source(self, instance_id: int,
                   extra_params: Optional[dict] = None) -> str:
        url = f'{self.base_url}/src'
        params = {"instance_id": instance_id}
        if extra_params is not None:
            for key in extra_params.keys():
                params[key] = extra_params[key]
        try:
            r = requests.get(url, params=params)
        except Exception as e:
            logger.error(f'Failed to request a source segment: {e}')
        return r.json()

    def send_hypo(self, instance_id: int, hypo: str) -> None:
        url = f'{self.base_url}/hypo'
        params = {"instance_id": instance_id}

        try:
            requests.put(url, params=params, data=hypo.encode("utf-8"))
        except Exception as e:
            logger.error(f'Failed to send a translated segment: {e}')

    def corpus_info(self):
        url = f'{self.base_url}'
        try:
            r = requests.get(url)
        except Exception as e:
            logger.error(f'Failed to request corpus information: {e}')

        return r.json()


def start_client(args):
    client = Client(args)
    client.reset_scorer()
    return client
