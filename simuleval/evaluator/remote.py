import logging
from tqdm import tqdm
from simuleval.data.segments import Segment, segment_from_json_string
import requests

logger = logging.getLogger("simuleval.remote_evaluator")


class RemoteEvaluator:
    def __init__(self, evaluator) -> None:
        self.evaluator = evaluator
        self.address = evaluator.args.remote_address
        self.port = evaluator.args.remote_port
        self.source_segment_size = evaluator.args.source_segment_size
        self.base_url = f"http://{self.address}:{self.port}"

    def send_source(self, segment: Segment):
        url = f"{self.base_url}/input"
        requests.put(url, data=segment.json())

    def receive_prediction(self) -> Segment:
        url = f"{self.base_url}/output"
        r = requests.get(url)
        return segment_from_json_string(r.text)

    def system_reset(self):
        requests.post(f"{self.base_url}/reset")

    def reset(self):
        self.evaluator.reset()

    def results(self):
        return self.evaluator.results()

    def remote_eval(self):
        for instance in tqdm(self.evaluator.instances.values()):
            self.system_reset()
            while not instance.finish_prediction:
                self.send_source(instance.send_source(self.source_segment_size))
                output_segment = self.receive_prediction()
                if not output_segment.is_empty:
                    instance.receive_prediction(output_segment)

        results = self.evaluator.results
        if self.evaluator.output:
            with open(self.evaluator.output / "scores", "w") as f:
                f.write(results)

        logger.info(f"Results:")
        print(results.to_string(index=False))
