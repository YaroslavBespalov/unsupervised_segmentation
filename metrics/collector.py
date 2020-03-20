import torch
import statistics
from torch import Tensor
from typing import List, Callable, Dict, Tuple


class MetricCollector:
    def __init__(self):
        self.name2func = {}
        self.name2values = {}
        self.preprocessors = {}

    def preprocessing(self, input: List[str], func: Callable[[Tuple[Tensor]], Tensor], name: str):
        self.preprocessors[name] = (input, func)

    def add_metric(self, input: List[str], func: Callable[[Tuple[Tensor]], Tensor], name: str):
        self.name2func[name] = (input, func)
        self.name2values[name] = []

    def add(self, input: Dict[str, Tensor]):
        with torch.no_grad():
            for preproc in self.preprocessors.keys():
                tmp_input, tmp_func = self.preprocessors[preproc]
                input[preproc] = tmp_func(*[input[i] for i in tmp_input])

            for key in self.name2func.keys():
                tmp_input, tmp_func = self.name2func[key]
                self.name2values[key].append(tmp_func(*[input[i] for i in tmp_input]).item())

    def clear_cash(self):
        self.name2values = {}
        for name in self.name2func.keys():
            self.name2values[name] = []

    def print_metrics(self):
        for name in self.name2func.keys():
            loss = statistics.mean(self.name2values[name])
            print(f"{name}: {loss}")






