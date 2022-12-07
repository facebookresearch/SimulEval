import json
from tqdm import tqdm
import requests

class RemoteScorer:
    def __init__(self, scorer, args) -> None:
        self.scorer = scorer
        self.address = args.eval_address
        self.port = args.eval_port
        self.source_segment_size = args.source_segment_size
        self.base_url = f"http://{self.address}:{self.port}"

    def send_source(self, segment):
        url = f"{self.base_url}/input"
        requests.put(url, data=json.dumps(segment))

    def receive_prediction(self):
        url = f"{self.base_url}/output"
        r = requests.get(url)
        return r.json()["output"]

    def system_reset(self):
        requests.post(f"{self.base_url}/reset")

    def reset(self):
        self.scorer.reset()

    def results(self):
        return self.scorer.results()

    def evaluate(self):
        for instance in tqdm(self.scorer.instances.values()):
            self.system_reset()
            while not instance.finish_prediction:
                self.send_source(instance.send_source(self.source_segment_size))
                output_segment_list = self.receive_prediction()
                for segment in output_segment_list:
                    instance.receive_prediction(segment)