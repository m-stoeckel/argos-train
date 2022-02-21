import csv
from collections import deque
from pathlib import Path

from argostrain.dataset import IDataset, trim_to_length_random


class CustomDataset(IDataset):
    def __init__(self, filepath, from_code='ka', to_code='en', has_header=True):
        path = Path(filepath)
        assert path.exists(), f"Path '{filepath}' does not exist"

        self.name = path.name.split('.', 1)[0]
        self.from_code = from_code
        self.to_code = to_code

        self.source = list()
        self.target = list()

        with path.open('r', encoding='utf-8', newline='') as fp:
            iterator = iter(csv.reader(fp, delimiter='\t'))

            # Skip header
            if has_header:
                header = next(iterator)
                from_idx = 0 if header[0] == from_code else 1
                to_idx = 1 if from_idx == 0 else 0
            else:
                from_idx = 0
                to_idx = 1

            for idx, row in enumerate(iterator, start=1):
                if len(row) < 2:
                    print(row)
                    raise ValueError("")
                source, target = row[from_idx], row[to_idx]
                self.source.append(source)
                self.target.append(target)
                self.size = idx

    def data(self, length=None):
        return trim_to_length_random(self.source, self.target, length)

    def __len__(self):
        return self.size
