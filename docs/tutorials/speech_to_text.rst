Speech-to-Text
==============

Whisper Agent
-----------------
Use whisper to evaluate custom audio for speech to text transcription.
First, change directory to :code:`speech_to_text`:

.. code-block:: bash
    cd examples/speech-to-text

Then, run the example code:

.. code-block:: bash
    simuleval \
    --agent whisper_waitk.py \
    --source-segment-size 500 \
    --waitk-lagging 3 \
    --source source.txt --target reference/transcript.txt \
    --output output --quality-metrics WER --visualize

The optional :code:`--visualize` tag generates N number of graphs in speech_to_text/output/visual directory where N corresponds to the number of source audio provided. An example graph can be seen `here <https://github.com/facebookresearch/SimulEval/pull/107>`_.

|
In addition, it supports the :code:`--score-only` command, where it will read data from :code:`instances.log` without running inference, which saves time if you just want the scores.

.. code-block:: bash
    simuleval --score-only --output output --visualize