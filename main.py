from ChatBot import My_ChatBot
from AE.AE_Bank import *

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = My_ChatBot(device=device, embed_dim=256)
    model.to(device)

    model.chat(True, True)


if __name__ == '__main__':
    main()