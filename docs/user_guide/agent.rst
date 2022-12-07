Agent
=====

To evaluate the simultaneous translation system,
the users need to implement agent class which operate the system logics.
This section will introduce how to implement an agent.

Source-Target Types
-------------------
First of all,
we must declare the source and target types of the agent class.
It can be done by inheriting from

- One of the following four built-in agent types

    - :class:`simuleval.agents.TextToTextAgent`
    - :class:`simuleval.agents.SpeechToTextAgent`
    - :class:`simuleval.agents.TextToSpeechAgent`
    - :class:`simuleval.agents.SpeechToSpeechAgent`

- Or :class:`simuleval.agents.GenericAgent`, with explicit declaration of :code:`source_type` and  :code:`target_type`.

The follow two examples are equivalent.

.. code-block:: python

    from simuleval import simuleval
    from simuleval.agents import GenericAgent

    class MySpeechToTextAgent(GenericAgent):
        source_type = "Speech"
        target_type = "Text"
        ....

.. code-block:: python

    from simuleval.agents import SpeechToSpeechAgent

    class MySpeechToTextAgent(SpeechToSpeechAgent):
        ....

.. _agent_policy:

Policy
------

The agent must have a :code:`policy` method which must return one of two actions, :code:`ReadAction` and :code:`WriteAction`.
For example, an agent with a :code:`policy` method should look like this

.. code-block:: python

    class MySpeechToTextAgent(SpeechToSpeechAgent):
        def policy(self):
            if do_we_need_more_input(self.states):
                return ReadAction()
            else:
                prediction = generate_a_token(self.states)
                finished = is_sentence_finished(self.states)
                return WriteAction(prediction, finished=finished)


..
    .. autoclass:: simuleval.agents.actions.WriteAction

..
    .. autoclass:: simuleval.agents.actions.ReadAction

States
------------
Each agent has the attribute the :code:`states` to keep track of the progress of decoding.
The :code:`states` attribute will be reset at the beginning of each sentence.
SimulEval provide an built-in states :class:`simuleval.agents.states.AgentStates`,
which has some basic attributes such source and target sequences.
The users can also define customized states with :code:`Agent.build_states` method:

.. code-block:: python

    from simuleval.agents.states import AgentStates
    from dataclasses import dataclass

    @dataclass
    class MyComplicatedStates(AgentStates)
        some_very_useful_variable: int

        def reset(self):
            super().reset()
            # also remember to reset the value
            some_very_useful_variable = 0

    class MySpeechToTextAgent(SpeechToSpeechAgent):
        def build_states(self):
            return MyComplicatedStates(0)

        def policy(self):
            some_very_useful_variable = self.states.some_very_useful_variable
            ...
            self.states.some_very_useful_variable = new_value
            ...

..
    .. autoclass:: simuleval.agents.states.AgentStates
        :members:


Pipeline
--------
The simultaneous system can consist several different components.
For instance, a simultaneous speech-to-text translation can have a streaming automatic speech recognition system and simultaneous text-to-text translation system.
SimulEval introduces the agent pipeline to support this function.
The following is a minimal example.
We concatenate two wait-k systems with different rates (:code:`k=2` and :code:`k=3`)
Note that if there are more than one agent class define,
the :code:`@entrypoint` decorator has to be used to determine the entry point

.. literalinclude:: ../../examples/quick_start/dummy_waitk_text_agent_v3.py
   :language: python
   :lines: 7-

Customized Arguments
-----------------------

It is often the case that we need to pass some customized arguments for the system to configure different settings.
The agent class has a built-in static method :code:`add_args` for this purpose.
The following is an updated version of the dummy agent from :ref:`first-agent`.

.. literalinclude:: ../../examples/quick_start/dummy_waitk_text_agent_v2.py
   :language: python
   :lines: 6-

Then just simply pass the arguments through command line as follow.

.. code-block:: bash

    simuleval \
        --source source.txt --source target.txt \ # data arguments
        --agent dummy_waitk_text_agent_v2.py \
        --waitk 3 --vocab data/dict.txt # agent arguments