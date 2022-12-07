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

logger = logging.getLogger("simuleval.server")


class SystemHandler(web.RequestHandler):
    def initialize(self, system):
        self.system = system

    def get(self):
        self.write(json.dumps({"info": str(self.system)}))


class ResetHandle(SystemHandler):
    def post(self):
        self.system.reset()


class OutputHandler(SystemHandler):
    def get(self):
        output = self.system.pop()
        if output is None:
            output = []
        r = json.dumps({"output": output})
        self.write(r)


class InputHandler(SystemHandler):
    def put(self):
        segment_info = json.loads(self.request.body)
        self.system.push(segment_info)


def start_service(system):
    app = web.Application(
        [
            (r"/reset", ResetHandle, dict(system=system)),
            (r"/input", InputHandler, dict(system=system)),
            (r"/output", OutputHandler, dict(system=system)),
            (r"/", SystemHandler, dict(system=system)),
        ],
        debug=False,
    )

    app.listen(12321, max_buffer_size=1024**3)

    logger.info(
        f"Evaluation Server Started (process id {os.getpid()}). Listening to port {12321} "
    )
    ioloop.IOLoop.current().start()
