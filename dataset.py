import torch
from torch.utils.data import Dataset

class CommentDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        raw_text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['class']
        tokens = raw_text.split()
        token_ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens[:self.max_len]]
        
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab['<pad>']] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        return {
            'input_ids': torch.tensor(token_ids, dtype = torch.long),
            'labels':  torch.tensor(label, dtype = torch.long)
        }