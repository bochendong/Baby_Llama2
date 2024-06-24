import torch
from Code.Utils.Read import read_config
from Code.Model.Model import Transformer
from Code.Train.Train import train_epoch
from Code.DataSet.Dataset import PretrainDataset
from Code.Tokenizer.Tokenizer import DataPreProcess

if __name__ == '__main__':
    config = read_config()
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
        num_workers=0
    )

    '''
    # Model
    model = Transformer(config["vocab_size"], config["n_layers"], config["n_heads"], 
                        config["max_seq_len"], config["embed_dim"], config["dropout"], 
                        config["norm_eps"], config["multiple_of"])
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(config["weight_decay"], config["learning_rate"], 
                                           (config["beta1"], config["beta2"]), 
                                           config["device"])
    raw_model = model

    # Train
    for epoch in range(config["max_epoch"]):
        train_epoch(epoch, model, raw_model, train_loader, optimizer, scaler,
                learning_rate = 3e-4, decay_lr = None, 
                gradient_accumulation_steps = 1, grad_clip = 1.0,
                device = 'cuda')

        torch.save(raw_model.state_dict(),f'Weight/epoch_{epoch}.pth')
'''

