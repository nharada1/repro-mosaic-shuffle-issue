from typing import Any
from ast import literal_eval as make_tuple

import torch
import tempfile

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from streaming import StreamingDataset

class FMOWStreamingDataset(StreamingDataset):
    def _process(self, obj):
        image = obj['image']
        dtype = obj['dtype']
        shape = make_tuple(obj['shape'])
        object = np.frombuffer(image, dtype=dtype)

        return object.reshape(shape)

    def __init__(self, local, remote, transform=None, **kwargs):
        super().__init__(local, remote, **kwargs)

        self.transform = transform

    def __getitem__(self, idx: int) -> Any:
        obj = super().__getitem__(idx)
    
        label = obj['class']
        filename = obj['filename']
        data = self._process(obj)

        if self.transform:
            data = self.transform(data)

        return data, label, filename


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Saving data to {tmp}")
        dataset = FMOWStreamingDataset(
            local=tmp,
            remote="/home/nharada/Datasets/mosaic/test_sets/fmow_full",
            split="val",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(256),
                    transforms.RandomCrop((256, 256)),
                ]
            ),
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
        )

        for idx, (data, label, filename) in enumerate(dataloader):
            print(idx)

if __name__ == "__main__":
    main()
