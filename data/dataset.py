import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self,
                 root='/',
                 csv_path='QA_data.csv',
                 embed_dim=768):
        super(MyDataset, self).__init__()

        self.data = pd.read_csv(csv_path)
        self.embed = np.load(root + 'embed_{}.npy'.format(embed_dim))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.embed[idx])
        return x

    def get_answer(self, idx=-1):
        if idx == -1:
            return 'None'
        else:
            return self.data["A"][idx]


def main():
    d = MyDataset()
    print(d.__getitem__(10).shape)
    print(d.get_answer(10))

if __name__ == "__main__":
    main()