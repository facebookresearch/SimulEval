"""Test script to calculate processing latency of multi-stream inputs.

Use it like this:

  python test_mult.py --file input.wav --num_streams 10 --use_vad

  Total running time 25.12
  Average start latency: 3.16
  Average finish latency: 1.24
"""

import math
import librosa
from argparse import ArgumentParser
from simuleval.data.segments import SpeechSegment, EmptySegment
from simuleval.utils import build_system_from_dir
from pathlib import Path
from src.vad import VAD
import time
import json
import numpy as np
class AudioFrontEnd:
    def __init__(self, samples, sample_rate, segment_size) -> None:
        self.samples = samples.tolist()
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.step = 0

    def get_segment(self):
        """
        This is the front-end logic in simuleval instance.py
        """

        num_samples = math.ceil(self.segment_size / 1000 * self.sample_rate)
        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                samples = self.samples[self.step :]
                is_finished = True
            else:
                samples = self.samples[self.step : self.step + num_samples]
                is_finished = False
            self.step = min(self.step + num_samples, len(self.samples))
            segment = SpeechSegment(
                index=self.step,
                content=samples,
                sample_rate=self.sample_rate,
                finished=is_finished,
            )
        else:
            # Finish reading this audio
            segment = EmptySegment(
                index=self.step,
                finished=True,
            )
        return segment
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--use_vad', action='store_true')
    parser.add_argument('--num_streams', type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    source_segment_size = 320  # milliseconds

    print("reading file")
    samples, sample_rate = librosa.load(args.file, sr=16000)
    print(f"finished reading file. {len(samples)} samples, {sample_rate} sample rate" )
    print("building system from dir")
    system = build_system_from_dir(
        Path(__file__).resolve().parent / "models/s2t_es-en_tt-waitk_multidomain"
    )
    print("finished building system from dir")


    runners = [{
        'id': i,
        "translations":[],
        "first_input_ts": None,
        "first_output_ts": None,
        "last_input_ts": None,
        "last_output_ts": None,
        "frontend": AudioFrontEnd(samples=samples, sample_rate=sample_rate, segment_size=source_segment_size,),
        "states": system.build_states(),
        "vad": VAD(),
        
    } for i in range(args.num_streams)]

    active = set(range(args.num_streams))
    print("STARTED LOOP")
    start_time = time.time()
    while len(active) > 0:
        if args.verbose:
            print("active", active)
        for runner in runners:
            
            input_segment = runner["frontend"].get_segment()
        
            if args.use_vad:
                # not doing actual work, but is useful for benchmarking performance
                np_arr = np.array(input_segment.content, dtype=np.float32)
                speech_probs = runner["vad"].get_speech_prob_from_np_float32(np_arr)
                if all(i <= 0.5 for i in speech_probs):
                    if args.verbose:
                        print("got silent chunk")
                else:
                    # got speech chunk
                    if args.verbose:
                        print("=======got speech chunk")

            output_segment = system.pushpop(input_segment, runner["states"])

            if runner['first_input_ts'] is None:
                runner['first_input_ts'] = time.time()

            if input_segment.finished and runner['last_input_ts'] is None:
                runner['last_input_ts'] = time.time()

            if output_segment.finished:
                if args.verbose:
                    print("finishing runner")
                runner["last_output_ts"] = time.time()
                active.remove(runner['id'])
            
            if not output_segment.is_empty:
                runner["translations"].append(output_segment.content)
                if runner['first_output_ts'] is None:
                    runner['first_output_ts'] = time.time()
                
    finish_time = time.time()
    res = [{
        "translations": " ".join(i["translations"]),
        "start_latency": i["first_output_ts"] - i["first_input_ts"],
        "finish_latency": i["last_output_ts"] - i["last_input_ts"],
    } for i in runners]

            
    if args.verbose:
        print(json.dumps(res, indent=4))
    print(f"Total running time {finish_time - start_time}")
    print("Average start latency:", round(sum(i["start_latency"] for i in res)/len(res), 2))
    print("Average finish latency:", round(sum(i["finish_latency"] for i in res)/len(res), 2))

if __name__ == "__main__":
    main()