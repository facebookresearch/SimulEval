import logging
import queue
import time
import torch
import numpy as np
import soundfile
from argparse import Namespace, ArgumentParser
from simuleval.agents import SpeechToSpeechAgent, AgentStates
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.data.segments import Segment, EmptySegment, SpeechSegment

logger = logging.getLogger()


class SileroVADStates(AgentStates):
    def __init__(self, args):
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        self.silence_limit_ms = args.silence_limit_ms
        self.window_size_samples = args.window_size_samples
        self.chunk_size_samples = args.chunk_size_samples
        self.sample_rate = args.sample_rate
        self.debug = args.debug
        self.test_input_segments_wav = None
        self.debug_log(args)
        self.input_queue: queue.Queue[Segment] = queue.Queue()
        self.next_input_queue: queue.Queue[Segment] = queue.Queue()
        super().__init__()

    def clear_queues(self):
        self.debug_log(f"clearing {self.input_queue.qsize()} chunks")
        while not self.input_queue.empty():
            self.input_queue.get_nowait()
            self.input_queue.task_done()
        self.debug_log(f"moving {self.next_input_queue.qsize()} chunks")
        # move everything from next_input_queue to input_queue
        while not self.next_input_queue.empty():
            chunk = self.next_input_queue.get_nowait()
            self.next_input_queue.task_done()
            self.input_queue.put_nowait(chunk)

    def reset(self) -> None:
        super().reset()
        # TODO: in seamless_server, report latency for each new segment
        self.first_input_ts = None
        self.silence_acc_ms = 0
        self.input_chunk = np.empty(0, dtype=np.int16)
        self.is_fresh_state = True
        self.clear_queues()
        self.model.reset_states()

    def get_speech_prob_from_np_float32(self, segment: np.ndarray):
        t = torch.from_numpy(segment)
        speech_probs = []
        # print("len(t): ", len(t))
        for i in range(0, len(t), self.window_size_samples):
            chunk = t[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                break
            speech_prob = self.model(chunk, self.sample_rate).item()
            speech_probs.append(speech_prob)
        return speech_probs

    def debug_log(self, m):
        if self.debug:
            logger.info(m)

    def process_speech(self, segment):
        """
        Process a full or partial speech chunk
        """
        queue = self.input_queue
        if self.source_finished:
            # current source is finished, but next speech starts to come in already
            self.debug_log("use next_input_queue")
            queue = self.next_input_queue

        # NOTE: we don't reset silence_acc_ms here so that once an utterance
        # becomes longer (accumulating more silence), it has a higher chance
        # of being segmented.
        # self.silence_acc_ms = 0

        if self.first_input_ts is None:
            self.first_input_ts = time.time() * 1000

        while len(segment) > 0:
            # add chunks to states.buffer
            i = self.chunk_size_samples - len(self.input_chunk)
            self.input_chunk = np.concatenate((self.input_chunk, segment[:i]))
            segment = segment[i:]
            self.is_fresh_state = False
            if len(self.input_chunk) == self.chunk_size_samples:
                queue.put_nowait(
                    SpeechSegment(content=self.input_chunk, finished=False)
                )
                self.input_chunk = np.empty(0, dtype=np.int16)

    def check_silence_acc(self):
        if self.silence_acc_ms >= self.silence_limit_ms:
            self.silence_acc_ms = 0
            if self.input_chunk.size > 0:
                # flush partial input_chunk
                self.input_queue.put_nowait(
                    SpeechSegment(content=self.input_chunk, finished=True)
                )
                self.input_chunk = np.empty(0, dtype=np.int16)
            self.input_queue.put_nowait(EmptySegment(finished=True))
            self.source_finished = True

    def update_source(self, segment: np.ndarray):
        speech_probs = self.get_speech_prob_from_np_float32(segment)
        chunk_size_ms = len(segment) * 1000 / self.sample_rate
        self.debug_log(
            f"{chunk_size_ms}, {len(speech_probs)} {[round(s, 2) for s in speech_probs]}"
        )
        window_size_ms = self.window_size_samples * 1000 / self.sample_rate
        if all(i <= 0.5 for i in speech_probs):
            if self.source_finished:
                return
            self.debug_log("got silent chunk")
            if not self.is_fresh_state:
                self.silence_acc_ms += chunk_size_ms
                self.check_silence_acc()
            return
        elif speech_probs[-1] <= 0.5:
            self.debug_log("=== start of silence chunk")
            # beginning = speech, end = silence
            # pass to process_speech and accumulate silence
            self.process_speech(segment)
            # accumulate contiguous silence
            for i in range(len(speech_probs) - 1, -1, -1):
                if speech_probs[i] > 0.5:
                    break
                self.silence_acc_ms += window_size_ms
            self.check_silence_acc()
        elif speech_probs[0] <= 0.5:
            self.debug_log("=== start of speech chunk")
            # beginning = silence, end = speech
            # accumulate silence , pass next to process_speech
            for i in range(0, len(speech_probs)):
                if speech_probs[i] > 0.5:
                    break
                self.silence_acc_ms += window_size_ms
            self.check_silence_acc()
            self.process_speech(segment)
        else:
            self.debug_log("======== got speech chunk")
            self.process_speech(segment)

    def debug_write_wav(self, chunk):
        if self.test_input_segments_wav is not None:
            self.test_input_segments_wav.seek(0, soundfile.SEEK_END)
            self.test_input_segments_wav.write(chunk)


class SileroVADAgent(SpeechToSpeechAgent):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.chunk_size_samples = args.chunk_size_samples
        self.args = args

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=float,
        )
        parser.add_argument(
            "--window-size-samples",
            default=512,  # sampling_rate // 1000 * 32 => 32 ms at 16000 sample rate
            type=int,
            help="Window size for passing samples to VAD",
        )
        parser.add_argument(
            "--chunk-size-samples",
            default=5120,  # sampling_rate // 1000 * 320 => 320 ms at 16000 sample rate
            type=int,
            help="Chunk size for passing samples to model",
        )
        parser.add_argument(
            "--silence-limit-ms",
            default=700,
            type=int,
            help="send EOS to the input_queue after this amount of silence",
        )
        parser.add_argument(
            "--debug",
            default=False,
            type=bool,
            help="Enable debug logs",
        )

    def build_states(self) -> SileroVADStates:
        return SileroVADStates(self.args)

    def policy(self, states: SileroVADStates):
        states.debug_log(
            f"queue size: {states.input_queue.qsize()}, input_chunk size: {len(states.input_chunk)}"
        )
        content = np.empty(0, dtype=np.int16)
        is_finished = states.source_finished
        while not states.input_queue.empty():
            chunk = states.input_queue.get_nowait()
            states.input_queue.task_done()
            content = np.concatenate((content, chunk.content))

        states.debug_write_wav(content)
        if is_finished:
            states.debug_write_wav(np.zeros(16000))

        if len(content) == 0:  # empty queue
            if not states.source_finished:
                return ReadAction()
            else:
                # NOTE: this should never happen, this logic is a safeguard
                segment = EmptySegment(finished=True)
        else:
            segment = SpeechSegment(
                content=content.tolist(),
                finished=is_finished,
                sample_rate=states.sample_rate,
            )

        return WriteAction(segment, finished=is_finished)
