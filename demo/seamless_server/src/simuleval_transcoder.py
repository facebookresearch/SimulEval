from simuleval.utils.agent import build_system_from_dir
from typing import Any, Tuple
import numpy as np
import soundfile
from fairseq.data.audio.audio_utils import convert_waveform
import io
import asyncio
from simuleval.data.segments import SpeechSegment, EmptySegment
import threading
import math
import logging
import sys
from pathlib import Path
import time
from g2p_en import G2p
import torch
import traceback
import time
import random

from .speech_and_text_output import SpeechAndTextOutput

MODEL_SAMPLE_RATE = 16_000

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))


class SimulevalTranscoder:

    def __init__(self, agent, sample_rate, debug, buffer_limit):
        self.agent = agent
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.states = self.agent.build_states()
        if debug:
            self.states[0].debug = True
        self.incoming_sample_rate = sample_rate
        self.close = False
        self.g2p = G2p()

        # buffer all outgoing translations within this amount of time
        self.output_buffer_idle_ms = 5000
        self.output_buffer_size_limit = buffer_limit # phonemes for text, seconds for speech
        self.output_buffer_cur_size = 0   
        self.output_buffer = []
        self.speech_output_sample_rate = None

        self.last_output_ts = time.time() * 1000
        self.timeout_ms = 30000  # close the transcoder thread after this amount of silence
        self.first_input_ts = None
        self.first_output_ts = None
        self.output_data_type = None   # speech or text
        self.debug = debug
        self.debug_ts = f'{time.time()}_{random.randint(1000, 9999)}'
        if self.debug:
            debug_folder = Path(__file__).resolve().parent.parent / "debug"
            self.test_incoming_wav = soundfile.SoundFile(
                debug_folder / f"{self.debug_ts}_test_incoming.wav",
                mode="w+",
                format="WAV",
                subtype="PCM_16",
                samplerate=self.incoming_sample_rate,
                channels=1
            )
            self.states[0].test_input_segments_wav = soundfile.SoundFile(
                debug_folder / f"{self.debug_ts}_test_input_segments.wav",
                mode="w+",
                format="WAV",
                samplerate=MODEL_SAMPLE_RATE,
                channels=1
            )

    def debug_log(self, *args):
        if self.debug:
            logger.info(*args)

    @classmethod
    def build_agent(cls, model_path):
        logger.info(f"Building simuleval agent: {model_path}")
        agent = build_system_from_dir(
            Path(__file__).resolve().parent.parent / f"models/{model_path}",
            config_name="vad_main.yaml"
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent.to(device, fp16=True)
        logger.info(f'Successfully built simuleval agent {model_path} on device {device}')

        return agent

    def process_incoming_bytes(self, incoming_bytes):
        segment, _sr = self._preprocess_wav(incoming_bytes)
        # # segment is array([0, 0, 0, ..., 0, 0, 0], dtype=int16)
        self.input_queue.put_nowait(segment)

    def get_input_segment(self):
        if self.input_queue.empty():
            return None
        chunk = self.input_queue.get_nowait()
        self.input_queue.task_done()
        return chunk

    def _preprocess_wav(self, data: Any) -> Tuple[np.ndarray, int]:
        segment, sample_rate = soundfile.read(
            io.BytesIO(data),
            dtype="float32",
            always_2d=True,
            frames=-1,
            start=0,
            format="RAW",
            subtype="PCM_16",
            samplerate=self.incoming_sample_rate,
            channels=1
        )
        if self.debug:
            self.test_incoming_wav.seek(0, soundfile.SEEK_END)
            self.test_incoming_wav.write(segment)

        segment = segment.T
        segment, new_sample_rate = convert_waveform(
            segment,
            sample_rate,
            normalize_volume=False,
            to_mono=True,
            to_sample_rate=MODEL_SAMPLE_RATE,
        )

        assert MODEL_SAMPLE_RATE == new_sample_rate
        segment = segment.squeeze(axis=0)
        return segment, new_sample_rate

    def process_pipeline_impl(self, input_segment):
        try:
            output_segment = self.agent.pushpop(input_segment, self.states)
            if self.states[0].first_input_ts is not None and self.first_input_ts is None:
                # TODO: this is hacky
                self.first_input_ts = self.states[0].first_input_ts

            if not output_segment.is_empty:
                self.output_queue.put_nowait(output_segment)

            if output_segment.finished:
                self.debug_log(
                    "OUTPUT SEGMENT IS FINISHED. Resetting states.")

                for state in self.states:
                    state.reset()

                if self.debug:
                    # when we rebuild states, this value is reset to whatever 
                    # is in the system dir config, which defaults debug=False.
                    self.states[0].debug = True
        except Exception as e:
            logger.error(f'Got exception while processing pipeline: {e}')
            traceback.print_exc()
        return input_segment

    def process_pipeline_loop(self):
        if self.close:
            return  # closes the thread

        self.debug_log("processing_pipeline")
        while not self.close:
            input_segment = self.get_input_segment()
            if input_segment is None:
                if self.states[0].is_fresh_state:  # TODO: this is hacky
                    time.sleep(0.3)
                else:
                    time.sleep(0.03)
                continue
            self.process_pipeline_impl(input_segment)
        self.debug_log("finished processing_pipeline")

    def process_pipeline_once(self):
        if self.close:
            return

        self.debug_log("processing pipeline once")
        input_segment = self.get_input_segment()
        if input_segment is None:
            return
        self.process_pipeline_impl(input_segment)
        self.debug_log("finished processing_pipeline_once")

    def get_output_segment(self):
        if self.output_queue.empty():
            return None

        output_chunk = self.output_queue.get_nowait()
        self.output_queue.task_done()
        return output_chunk

    def start(self):
        self.debug_log("starting transcoder in a thread")
        threading.Thread(target=self.process_pipeline_loop).start()
    
    def first_translation_time(self):
        return round((self.first_output_ts - self.first_input_ts) / 1000,  2)

    def get_buffered_output(self) -> SpeechAndTextOutput:
        now = time.time() * 1000
        self.debug_log(f'get_buffered_output queue size: {self.output_queue.qsize()}')
        while not self.output_queue.empty():            
            tmp_out = self.get_output_segment()
            if tmp_out and len(tmp_out.content) > 0:
                if not self.output_data_type: self.output_data_type = tmp_out.data_type
                if len(self.output_buffer) == 0:
                    self.last_output_ts = now
                self._populate_output_buffer(tmp_out)
                self._increment_output_buffer_size(tmp_out)
                
                if tmp_out.finished:
                    res = self._gather_output_buffer_data(final=True)
                    self.output_buffer = []
                    self.increment_output_buffer_size = 0
                    self.last_output_ts = now
                    self.first_output_ts = now
                    return res

        if len(self.output_buffer) > 0 and (
            now - self.last_output_ts >= self.output_buffer_idle_ms or
            self.output_buffer_cur_size >= self.output_buffer_size_limit
        ):
            self.last_output_ts = now
            res = self._gather_output_buffer_data(final=False)
            self.output_buffer = []
            self.output_buffer_phoneme_count = 0
            self.first_output_ts = now
            return res
        else:
            return None
              
    def _gather_output_buffer_data(self, final):
        if self.output_data_type == "text":
            return SpeechAndTextOutput(text=" ".join(self.output_buffer), final=final)
        elif self.output_data_type == "speech":
            return SpeechAndTextOutput(
                speech_samples=self.output_buffer, 
                speech_sample_rate=MODEL_SAMPLE_RATE,
                final=final
            )
        else:
            raise ValueError(f"Invalid output buffer data type: {self.output_data_type}")

    def _increment_output_buffer_size(self, segment):
        if segment.data_type == "text":
            self.output_buffer_cur_size += self._compute_phoneme_count(segment.content)
        elif segment.data_type == "speech":
            self.output_buffer_cur_size += len(segment.content) / MODEL_SAMPLE_RATE  # seconds

    def _populate_output_buffer(self, segment):
        if segment.data_type == "text":
            self.output_buffer.append(segment.content)
        elif segment.data_type == "speech":
            self.output_buffer += segment.content
        else:
            raise ValueError(f"Invalid segment data type: {segment.data_type}")
        
    def _compute_phoneme_count(self, string: str) -> int:
        return len([x for x in self.g2p(string) if x != " "])
