import torch
from MyModule.ChatBot import My_ChatBot

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = My_ChatBot(base_path='data/', device=device)
    model.to(device)

    model.chat(True, True)


if __name__ == '__main__':
    main()