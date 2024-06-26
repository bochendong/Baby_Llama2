import torch
import pandas as pd
from Code.Utils.Read import read_config
from Code.Utils.Logging import setup_logging
from Code.Model.Model import Transformer
from Code.Train.Train import train_epoch
from Code.SFT.DataClean.CleanAlpacaGpt4 import cleanAlpacaGpt4File
from Code.DataSet.Dataset import PretrainDataset
from Code.Tokenizer.Tokenizer import DataPreProcess


if __name__ == '__main__':
    config = read_config()
    setup_logging("./Log/training.log")

    SFT = True
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

    # Train
    for epoch in range(config["max_epoch"]):
        train_epoch(epoch, model, raw_model, train_loader, optimizer, scaler,
                learning_rate = 3e-4, decay_lr = None, 
                gradient_accumulation_steps = 1, grad_clip = 1.0,
                device = config["device"])

        torch.save(raw_model.state_dict(),f'Weight/epoch_{epoch}.pth')

    '''
    if SFT == True:
        df = pd.DataFrame(columns=['prompt', 'answer'])
        questions, answers =  cleanAlpacaGpt4File
        df['prompt'] = questions
        df['answer'] = answers
        df.to_csv('./data/SFT/sft_data.csv', index=False)
    '''
