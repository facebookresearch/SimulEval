## to start the server

Clone the repo
`cd seamless-experiences/seamless_vc/seamless_server`

If running for the first time, create conda environment from the environment.yaml `conda env create -f environment.yml`
(or if you are on Mac OS, replace `environment.yml` with `environment_mac.yml`)

In each new terminal you use you will need to activate the conda environment:
`conda activate smlss_server`

To install Seamless related code run:
`pip install git+ssh://git@github.com/facebookresearch/SimulEval.git`
`pip install git+ssh://git@github.com/fairinternal/fairseq-py.git@seamless_main`

Run the server:
`pyhton app.py`

## Setup Google account if not already set up

https://cloud.google.com/translate/media/docs/streaming
Get google credential and put it into google_credentials.json file in the root of the repo.

## Download the Seamless models

See [list of current available demo](https://www.internalfb.com/intern/wiki/FAIR_Accel_Language/Projects/Seamless/Workstreams/Streaming/Tutorials/Demo/#available-models) in FAIR Seamless wiki. We need some following models to make the server run:

- es->en s2t model: Put the checkpoint file in "models/s2t_es-en_emma_multidomain_v0.1" under the root directory. To get the model files contact researchers ([Anna Sun](https://www.internalfb.com/profile/view/1115461094) or [Xutai Ma](https://www.internalfb.com/profile/view/100004735920998)).

## Debuging

For start_seamless_stream_es_en_s2t endpoint you can set debug=true when sending config event.
This enables extensive debug logging and it saves audio files in /debug folder. test_no_silence.wav contains data with silence chunks removed.
