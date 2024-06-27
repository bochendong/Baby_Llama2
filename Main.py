import os
import torch
import pandas as pd
from Code.Utils.Read import read_config
from Code.Utils.Logging import setup_logging

from Code.Tokenizer.Tokenizer import DataPreProcess
from Code.Tokenizer.GLMTokenizer import ChatGLMTokenizer

from Code.Model.Model import Transformer
from Code.Train.Train import train_epoch

from Code.DataSet.Dataset import PretrainDataset
from Code.SFT.DataSet.SFTDataSet import SFTDataset
from Code.SFT.Train.TrainSFT import train_SFT_epoch
from Code.SFT.DataClean.CleanAlpacaGpt4 import cleanAlpacaGpt4File

if __name__ == '__main__':
    config = read_config()
    setup_logging("./Log/training.log")

    SFT = False
    dtype = 'float16'

    DataPreProcess()
    data_path_list=[
        './data/pretrain_data.bin'
    ]

    # Dataset Preparation
    train_ds = PretrainDataset(data_path_list, max_length=config["max_seq_len"],use_memmap=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"],
        pin_memory=False, drop_last=False, shuffle=False,        
        num_workers=0 if config["device"] == 'cpu' else 4
    )

    # Model
    model = Transformer(config)
    model.to(config["device"])
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(config["weight_decay"], config["learning_rate"], 
                                           (config["beta1"], config["beta2"]), config["device"])
    raw_model = model

    # Train Model
    if (os.path.exists(f'Weight/epoch_{config["max_epoch"] - 1}.pth') == False):
        for epoch in range(config["max_epoch"]):
            train_epoch(epoch, model, raw_model, train_loader, optimizer, scaler,
                    learning_rate = 3e-4, decay_lr = None, 
                    gradient_accumulation_steps = 1, grad_clip = 1.0,
                    device = config["device"])

            torch.save(raw_model.state_dict(),f'Weight/epoch_{epoch}.pth')
    
    # Fine tune
    if SFT == True:
        if (os.path.exists('./data/SFT/sft_data.csv') == False):
            df = pd.DataFrame(columns=['prompt', 'answer'])
            questions, answers = cleanAlpacaGpt4File()
            df['prompt'], df['answer'] = questions, answers
            df.to_csv('./data/SFT/sft_data.csv', index=False)
        
        df = pd.read_csv('./data/SFT/sft_data.csv')
        tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
        train_ds = SFTDataset(df, tokenizer,max_length=256)

        model = model.load_state_dict(torch.load(f'./Weight/epoch_{config["max_epoch"] - 1}.pth'))
        raw_model = model
        model.to(config["device"])
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1,
                        pin_memory=False, drop_last=False, shuffle=False,num_workers=0)

        for epoch in range(config["max_epoch"]):
            train_SFT_epoch(epoch, model, train_loader, optimizer, scaler,
                    learning_rate = 3e-4, decay_lr = None, grad_clip = 1.0,
                    device = config["device"])
            torch.save(raw_model.state_dict(),f'Weight/SFT_epoch_{epoch}.pth')


