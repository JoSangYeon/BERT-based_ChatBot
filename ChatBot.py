import time
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from data.dataset import MyDataset
from AE.AE_Bank import *


class My_ChatBot(nn.Module):
    def __init__(self, model_path='jhgan/ko-sroberta-multitask',
                 csv_path='data/QA_data.csv',
                 embed_dim=768,
                 batch_size=512,
                 device='cpu'):
        super(My_ChatBot, self).__init__()

        # settings values #
        self.device = device
        self.batch_size = batch_size
        self.user_embed = 0

        # settings data #
        self.dataset = MyDataset(root='data/', csv_path=csv_path, embed_dim=embed_dim)
        self.loader = DataLoader(self.dataset, batch_size=batch_size)

        # settings forward module #
        self.bert = SentenceTransformer(model_path)
        if embed_dim==768:
            self.AE = nn.Sequential()
        else:
            self.AE = torch.load('AE/AE' + str(embed_dim) + '.pt').encoder
        self.cos = nn.CosineSimilarity(dim=-1)


    def set_user_embed(self, text):
        embed = self.sub_forward(text)
        embed = torch.tensor(embed)
        embed = embed.view(1, -1).to(self.device) #(1, dim)
        embed = self.AE(embed)
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

            f = time.time()
            answer, sim, _ = self.inference(self.loader, user)
            f = time.time() - f

            print(" BOT >>>", answer)
            if s : print("\t유사도 : {:.4f}%".format(sim * 100))
            if t : print("\t추론 시간 :", f)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = My_ChatBot(device=device, embed_dim=64)
    model.to(device)

    model.chat(True, True)


if __name__ == "__main__":
    main()