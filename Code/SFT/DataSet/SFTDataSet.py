import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split


class SFTDataset(Dataset):
    def __init__(self,df,tokenizer, max_length=256, prompt_max_len=128
                 , answer_max_len=128):
        super().__init__()
        self.df=df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.special_tokens['<bos>']
        self.eos = self.tokenizer.special_tokens['<eos>']
        self.pad = 0
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(sample['prompt'],add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'],add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len-2]

        input_id = prompt + [self.bos] + answer + [self.eos]
        context_length = input_id.index(self.bos)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        if pad_len==0:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len

        input_id=np.array(input_id)

        X=np.array(input_id[:-1]).astype(np.int64)
        Y=np.array(input_id[1:]).astype(np.int64)
        loss_mask=np.array(loss_mask[:-1])
        return torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(loss_mask)