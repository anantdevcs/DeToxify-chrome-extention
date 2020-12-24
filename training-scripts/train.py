import torch
import torch.nn as nn
from model import RNNClassifier
from loader import get_loader
from tqdm import tqdm 
from tqdm import trange

def train(df_dir, num_epochs, train_dl, train_ds, rnn_model, device, criterion, optimizer):
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
    print(f"Found {device}")
    rnn_model.train()
    rnn_model = rnn_model.to(device)
    train_loss_list = []
    train_acc_list = []
#     tr = trange(num_epochs, desc='Desc appear here', leave=True)


    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        num_ex = 0
        for sample, target in tqdm(train_dl):
            sample, target = sample.to(device), target.to(device)
            pred = rnn_model(sample)
            loss = criterion(pred, target)        
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            num_ex += 1
            epoch_loss += loss.item()
            epoch_acc +=  (torch.softmax(pred, dim=1).argmax(dim=1) == target).sum().float() / float( target.size(0) )
            
        
        print(f"train epoch [{epoch}] acc {epoch_acc / num_ex * 100}")
        print(f"train epoch [{epoch}] loss {epoch_loss / num_ex } ")
        train_loss_list.append(epoch_loss / num_ex)
        train_acc_list.append(epoch_acc / num_ex * 100)


        
#         tr.set_description(desc=f'Loss {epoch_loss:.3f}')

    
def predict_sentiment(rnn_model, text, train_vocab):
        numeric = train_vocab.numericalize(text).unsqueeze(0)
        print(numeric)
        pred = rnn_model(numeric)
        print(f"Prediction {pred}")
        return torch.argmax(pred)



train('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip',30, train_dl, train_ds, rnn_model, device, criterion, optimizer)
        


# train(df_dir='../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

