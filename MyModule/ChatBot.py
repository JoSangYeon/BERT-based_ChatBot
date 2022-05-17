import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import notebook, tqdm
from sentence_transformers import SentenceTransformer

from MyModule.dataset import MyDataset


class My_ChatBot(nn.Module):
    def __init__(self, model_path='jhgan/ko-sroberta-multitask',
                 base_path='data/',
                 csv_path='QA_data.csv',
                 embed_dim=768,
                 batch_size=512,
                 device='cpu'):
        super(My_ChatBot, self).__init__()

        # settings values #
        self.device = device
        self.batch_size = batch_size
        self.user_embed = 0

        # settings data #
        self.dataset = MyDataset(base_path=base_path,csv_path=csv_path, embed_dim=embed_dim)
        self.loader = DataLoader(self.dataset, batch_size=batch_size)

        # settings forward module #
        self.bert = SentenceTransformer(model_path)
        self.cos = nn.CosineSimilarity(dim=-1)


    def set_user_embed(self, text):
        embed = self.sub_forward(text)
        embed = torch.tensor(embed)
        embed = embed.view(1, -1).to(self.device) #(1, dim)
        self.user_embed = embed


    def sub_forward(self, x):
        x = self.bert.encode(x)
        return x


    def forward(self, x):
        sim = self.cos(self.user_embed, x) # Compare (1, dim) to (batch, dim) => (batch,) : Similarity
        return sim


    def get_max(self, sim, max_sim, max_idx, batch_idx):
        v, i = torch.max(sim, dim=-1)
        if max_sim < v.item():
            max_sim = v.item()
            max_idx = i.item() + (self.batch_size * batch_idx)

        return max_sim, max_idx


    def inference(self, loader, text):
        self.eval()
        self.set_user_embed(text)
        max_sim = 0
        max_idx = 0

        with torch.no_grad():
            for b_idx, data in enumerate(loader):
                data = data.to(self.device)

                batch_sim = self(data)

                max_sim, max_idx = self.get_max(batch_sim, max_sim, max_idx, b_idx)
        answer = self.dataset.get_answer(max_idx)
        return answer, max_sim, max_idx


    def chat(self, s=False, t=False):
        """
        Ultimately, the method used by the user
        :param s: Whether to output similarity
        :param t: Whether to output the output time
        """
        while (True):
            user = input("USER >>> ")
            if user == 'exit' or user == 'quit':
                break

            s = time.time()
            answer, sim, _ = self.inference(self.loader, user)
            e = time.time()

            print(" BOT >>>", answer)
            if s : print("\t유사도 : {:.4f}%".format(sim * 100))
            if t : print("\t추론 시간 :", e - s)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = My_ChatBot(base_path='../data/', device=device)
    model.to(device)

    model.chat(True, True)


if __name__ == "__main__":
    main()