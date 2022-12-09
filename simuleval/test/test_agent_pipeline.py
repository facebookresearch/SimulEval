import subprocess
import random
from simuleval.agents import TextToTextAgent
from simuleval.agents import AgentPipeline
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import TextSegment

def test_pipeline_cmd():
    result = subprocess.Popen(
        [
            "simuleval",
            "--agent", "examples/quick_start/agent_pipeline.py",
            "--source", "examples/quick_start/source.txt",
            "--target", "examples/quick_start/target.txt",
            ]
    )
    _ = result.communicate()[0]
    returncode = result.returncode
    assert returncode == 0

def test_pipeline():

    class DummyWaitkTextAgent(TextToTextAgent):
        waitk = 0
        vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

        def policy(self):
            lagging = len(self.states.source) - len(self.states.target)

            if lagging >= self.waitk or self.states.source_finished:
                prediction = self.vocab[len(self.states.source)]

                return WriteAction(prediction, finished=(lagging <= 1))
            else:
                return ReadAction()


    class DummyWait2TextAgent(DummyWaitkTextAgent):
        waitk = 2


    class DummyWait4TextAgent(DummyWaitkTextAgent):
        waitk = 4


    class DummyPipeline(AgentPipeline):
        pipeline = [DummyWait2TextAgent, DummyWait4TextAgent]

    args = None
    agent_1 = DummyPipeline.from_args(args)
    agent_2 = DummyPipeline.from_args(args)
    for _ in range(10):
        segment = TextSegment(0, "A")
        output_1 = agent_1.pushpop(segment)
        agent_2.push(segment)
        output_2 = agent_2.pop()
        assert output_1.content == output_2.content


