FROM python:3.8
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN pip install -U openai-whisper
RUN pip install -U editdistance
RUN git clone https://github.com/facebookresearch/SimulEval
WORKDIR /SimulEval/
RUN pip install -e .
WORKDIR /SimulEval/examples/speech_to_text/
#CMD ["simuleval", "--agent", "whisper_waitk.py", "--source-segment-size", "500", "--waitk-lagging", "3", "--source", "source.txt", "--target", "reference/transcript.txt", "--output", "output", "--quality-metrics", "WER"]
CMD ["simuleval", "--standalone", "--remote-port", "8888", "--agent", "whisper_waitk.py",  "--waitk-lagging", "3"]
