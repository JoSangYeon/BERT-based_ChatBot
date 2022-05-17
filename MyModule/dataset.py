import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, base_path='data/',
                 csv_path='QA_data.csv',
                 embed_dim=768):
        super(MyDataset, self).__init__()

        self.data = pd.read_csv(base_path+csv_path)
        self.embed = np.load(base_path+'embed_{}.npy'.format(embed_dim))

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
    d = MyDataset(base_path='../data/')
    print(d.__getitem__(10).shape)
    print(d.get_answer(10))

if __name__ == "__main__":
    main()