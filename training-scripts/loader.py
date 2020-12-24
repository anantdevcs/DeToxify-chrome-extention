from pandas.core.tools import numeric
import torch
import torch.nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import spacy
from tqdm import tqdm
spacy_eng = spacy.load("en")


class Vocab:

    def __init__(self, freq_threshold=2):
        self.itos = {0:'<PAD>', 1:'<SOS>', 2:'<EOS>', 3:'<UNK>'}
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.freq_threshold = freq_threshold


    def __getitem__(self, index):
        return 

    def build_vocab(self, sentence_list):
        freq = {}
        idx = 4
        print(f'Started Tokenizing --- FOUND {len(sentence_list)} sentences')
        for sentence in tqdm(sentence_list):
            for word in self.tokenize_text(sentence):
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1
                if freq[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                

    def vocab_size(self):
        return len(self.itos)
    
    @staticmethod
    def tokenize_text(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def numericalize(self, text):
        tokenized_text = self.tokenize_text(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# [seq_len, batch, emb]



class ToxicDataset(torch.utils.data.Dataset):
    """Some Information about ToxicDataset"""
    def __init__(self, df_dir, freq_threshold=2, frac_use=0.01):
        super(ToxicDataset, self).__init__()
        self.df_dir = df_dir
        self.df = pd.read_csv(self.df_dir)
        self.vocab = Vocab(freq_threshold=freq_threshold)
        self.sentences = self.df['comment_text'][0:int(frac_use*len(self.df))]
        self.targets = self.df['toxic'][0:int(frac_use*len(self.df))]
        self.vocab.build_vocab(self.sentences.tolist())

    def __getitem__(self, index):
        text = self.sentences[index]
        target = self.targets[index]
        numeric_cap = [self.vocab.stoi['<SOS>']]
        numeric_cap += (self.vocab.numericalize(text))
        numeric_cap.append(self.vocab.stoi['<EOS>'])

        return torch.tensor(numeric_cap), torch.tensor(target) 

    def __len__(self):
        return len(self.df) 


class MyCollate:
    def __init__(self, pad_idx=4):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        texts = [item[0].unsqueeze(1) for item in batch ]
#         print(texts[0].shape, texts[1].shape)
        texts = pad_sequence(texts, padding_value=self.pad_idx)
#         print(texts.shap)
#         texts = torch.cat(texts, dim=0)
        targets = [item[1] for item in batch]
        
        return texts, torch.tensor(targets)
        


def get_loader(df_dir, batch_size=32, num_workers=4, freq_threshold=2):
    toxic_ds = ToxicDataset(df_dir, freq_threshold=freq_threshold)
    dataloader = torch.utils.data.DataLoader(toxic_ds, batch_size=batch_size, shuffle=False, collate_fn=MyCollate())
    return dataloader, toxic_ds

# dl, ds = get_loader('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
# for text, tar in dl:
#     print(text.shape)
#     print(tar.shape)
#     break

