import os

# import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from Code.Utils.Read import read_config
from Code.Utils.Logging import setup_logging

from Code.Tokenizer.Tokenizer import DataPreProcess
# from Code.Tokenizer.GLMTokenizer import ChatGLMTokenizer

from Code.Model.Model import Transformer
from Code.Train.Train import train_epoch

from Code.DataSet.Dataset import PretrainDataset
# from Code.SFT.DataSet.SFTDataSet import SFTDataset
# from Code.SFT.Train.TrainSFT import train_SFT_epoch
# from Code.SFT.DataClean.CleanAlpacaGpt4 import cleanAlpacaGpt4File

import torch.nn as nn
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def init_process(rank, num_gpus, train_fn, config, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend, rank=rank, world_size=num_gpus)
    train_fn(rank, num_gpus, config)

def check_available_gpus():
    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available.")
        else:
            print(f"Number of available GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        return num_gpus
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure PyTorch is installed with ROCm support.")

def train(rank, num_gpus, config):
    torch.manual_seed(0)

    if (num_gpus == 0):
        device = torch.device(f'cpu')
    elif (num_gpus == 1):
        device = torch.device(f'cuda')
    else:
        device = torch.device(f'cuda:{rank}')
    '''
    print(f"Device: {device}")
    data_path_list = ['/lustre/orion/bif146/world-shared/enzhi/baby_llama/Baby_Llama2/data/pretrain_data.bin']


    train_ds = PretrainDataset(data_path_list, max_length=config["max_seq_len"], use_memmap=True)
    if (num_gpus > 1):
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=num_gpus, rank=rank)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"],
        pin_memory=False, drop_last=False, shuffle=False,
        num_workers=0 if num_gpus == 0 else 4,
        sampler=train_sampler
    )
    '''

    model = SimpleNN().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    '''
    if (num_gpus > 1):
        model = DDP(model, device_ids=[rank])
        raw_model = model.module
    else:
        raw_model = model

    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    optimizer = model.configure_optimizers(config["weight_decay"], config["learning_rate"], 
                                                      (config["beta1"], config["beta2"]), config["device"])

    if not os.path.exists(f'Weight/epoch_{config["max_epoch"] - 1}.pth'):
        for epoch in range(config["max_epoch"]):
            train_epoch(epoch, raw_model, raw_model, train_loader, optimizer, scaler,
                        learning_rate=3e-4, decay_lr=None,
                        gradient_accumulation_steps=1, grad_clip=1.0,
                        device=device)
            if rank == 0:
                torch.save(raw_model.state_dict(), f'Weight/epoch_{epoch}.pth')

    SFT = config["SFT"]

    if SFT:
        if not os.path.exists('./data/SFT/sft_data.csv'):
            df = pd.DataFrame(columns=['prompt', 'answer'])
            questions, answers = cleanAlpacaGpt4File()
            df['prompt'], df['answer'] = questions, answers
            df.to_csv('./data/SFT/sft_data.csv', index=False)
        
        df = pd.read_csv('./data/SFT/sft_data.csv')
        tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
        train_ds = SFTDataset(df, tokenizer, max_length=256)

        ddp_model.load_state_dict(torch.load(f'./Weight/epoch_{config["max_epoch"] - 1}.pth'))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=num_gpus, rank=rank)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, pin_memory=False, drop_last=False, 
                                                   shuffle=False, num_workers=0, sampler=train_sampler)

        for epoch in range(config["max_epoch"]):
            train_SFT_epoch(epoch, ddp_model, train_loader, optimizer, scaler,
                            learning_rate=3e-4, decay_lr=None, grad_clip=1.0,
                            device=device)
            if rank == 0:
                torch.save(raw_model.state_dict(), f'Weight/SFT_epoch_{epoch}.pth')'''
    

if __name__ == '__main__':
    dir_path = '/lustre/orion/bif146/world-shared/enzhi/baby_llama/Baby_Llama2'

    config = read_config()

    if config["Preprocess"] == True:
        DataPreProcess()
    
    setup_logging(dir_path + "/Log/training.log")
    num_gpus = check_available_gpus()

    if(num_gpus > 1):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        processes = []
        for rank in range(num_gpus):
            p = torch.multiprocessing.Process(target=init_process, args=(rank, num_gpus, train, config))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        train(rank = 0, num_gpus = 1, config = config)

