Running the demo:
1. Create a directory for the dummy model: `models/$DUMMY_MODEL`
2. Create a new yaml file `models/$DUMMY_MODEL/vad_main.yaml`, with the following:
```
agent_class: examples.speech_to_text_demo.english_counter_pipeline.EnglishCounterAgentPipeline
```
3. Set the available agent in `SimulevalAgentDirectory.py` to `$DUMMY_MODEL`
4. Run `python app.py`


- Note: If you get an ImportError for `examples.speech_to_text_demo`, run `python -c "import examples; print(examples.__file__)"`. If the file is something like `$PREFIX/site-packages/examples/__init__.py`, run `rm -r $PREFIX/site-packages/examples` and try again.