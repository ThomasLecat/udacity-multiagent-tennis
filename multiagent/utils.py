import csv
from typing import Any, List, TextIO, Union

import torch

from ccontrol.replay_buffer import SampleBatch, TorchSampleBatch


def convert_to_torch(
    sample_batch: SampleBatch, device: torch.device
) -> TorchSampleBatch:
    return TorchSampleBatch(
        **{
            key: torch.from_numpy(value).to(device)
            for key, value in sample_batch._asdict().items()
        }
    )


def listify(x: Union[List, Any]) -> List:
    if type(x) != list:
        return [x]
    return x


def write_list_to_csv(file: TextIO, values: List) -> None:
    csv_writer = csv.writer(file, delimiter=",")
    for value in values:
        csv_writer.writerow(listify(value))
