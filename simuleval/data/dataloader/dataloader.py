from simuleval import SUPPORTED_TARGET_MEDIUM, SUPPORTED_SOURCE_MEDIUM
from importlib.resources import path
from typing import Any, List, Tuple, Union


class GenericDataloader:
    def __init__(self, source_list: Union[path, str], target_list: List[str]) -> None:
        self.source_list = source_list
        self.target_list = target_list
        assert len(self.source_list) == len(self.target_list)

    def get_source(self, index: int) -> List:
        return self.preprocess_source(self.source_list[index])

    def get_target(self, index: int) -> List:
        return self.preprocess_source(self.source_list[index])

    def __getitem__(self, index: int) -> Tuple[List, List]:
        return self.get_source(i), self.get_target(j)

    def preprocess_source(self, source: str) -> Any:
        raise NotImplementedError

    def preprocess_target(self, target: str) -> Any:
        raise NotImplementedError
