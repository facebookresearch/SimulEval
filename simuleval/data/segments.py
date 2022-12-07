from dataclasses import dataclass, field


@dataclass
class Segment:
    index: int = 0
    content: list = field(default_factory=list)
    finished: bool = False
    is_empty: bool = False


@dataclass
class EmptySegment(Segment):
    is_empty: bool = True


@dataclass
class TextSegment(Segment):
    content: str = ""


@dataclass
class SpeechSegment(Segment):
    sample_rate: int = -1
