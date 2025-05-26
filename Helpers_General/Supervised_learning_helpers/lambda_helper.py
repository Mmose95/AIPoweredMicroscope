import torch.nn as nn

class TupleSelect(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        print(f"[TupleSelect] Extracting index {self.index} from {type(x)}")
        return x[self.index]