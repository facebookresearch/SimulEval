# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import json
import logging
import argparse
from tornado import web, ioloop
import simuleval
from simuleval.scorer import Scorer

DEFAULT_HOSTNAME = 'localhost'
DEFAULT_PORT = 12321


logger = logging.getLogger('simuleval.server')


class ScorerHandler(web.RequestHandler):
    def initialize(self, scorer):
        self.scorer = scorer


class EvalSessionHandler(ScorerHandler):
    def post(self):
        self.scorer.reset()

    def get(self):
        r = json.dumps(self.scorer.get_info())
        self.write(r)


class ResultHandler(ScorerHandler):
    def get(self):
        instance_id = self.get_argument('instance_id', None)

        if instance_id is not None:
            instance_id = int(instance_id)
            instance = self.scorer.instances[instance_id]
            r = json.dumps(instance.summarize())

        else:
            r = json.dumps(self.scorer.score())

        self.write(r)


class SourceHandler(ScorerHandler):
    def get(self):
        instance_id = int(self.get_argument('instance_id'))
        segment_size = None
        if "segment_size" in self.request.arguments:
            string = self.get_argument('segment_size')
            if len(string) > 0:
                segment_size = int(string)

        r = json.dumps(self.scorer.send_src(int(instance_id), segment_size))

        self.write(r)


class HypothesisHandler(ScorerHandler):
    def put(self):
        instance_id = int(self.get_argument('instance_id'))
        list_of_tokens = self.request.body.decode('utf-8').strip().split()
        self.scorer.recv_hyp(instance_id, list_of_tokens)


def start_server(args):
    scorer = Scorer(args)

    app = web.Application([
        (r'/result', ResultHandler, dict(scorer=scorer)),
        (r'/src', SourceHandler, dict(scorer=scorer)),
        (r'/hypo', HypothesisHandler, dict(scorer=scorer)),
        (r'/', EvalSessionHandler, dict(scorer=scorer)),
    ], debug=False)

    app.listen(args.port, max_buffer_size=1024 ** 3)

    logger.info(f"Evaluation Server Started (process id {os.getpid()}). Listening to port {args.port} ")
    ioloop.IOLoop.current().start()


def main():
    parser = argparse.ArgumentParser()
    simuleval.options.add_server_args(parser)
    simuleval.options.add_data_args(parser)
    args = parser.parse_args()

    if args.data_type is None:
        sys.exit(
            "Data type is needed, set it by --data-type or env SIMULEVAL_DATA_TYPE")

    start_server(args)


if __name__ == '__main__':
    main()
