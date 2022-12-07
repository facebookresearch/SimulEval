from simuleval.data.segments import Segment, TextSegment


class AgentStates:
    """
    Tracker of the decoding progress.

    Attributes:
        source (list): current source sequence.
        target (list): current target sequence.
        source_finished (bool): if the source is finished.
        target_finished (bool): if the target is finished.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset Agent states"""
        self.source = []
        self.target = []
        self.source_finished = False
        self.target_finished = False

    def update_source(self, segment: Segment):
        """
        Update states from input segment

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if not self.source_finished:
            if isinstance(segment, TextSegment):
                self.source.append(segment.content)
            else:
                self.source += segment.content

    def update_target(self, segment: Segment):
        """
        Update states from output segment

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.target_finished = segment.finished
        if not self.target_finished:
            if isinstance(segment, TextSegment):
                self.target.append(segment.content)
            else:
                self.target += segment.content
