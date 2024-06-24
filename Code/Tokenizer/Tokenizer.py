import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from Code.Tokenizer.GLMTokenizer import ChatGLMTokenizer

DataPath = './data'
FileName = {
    "Wiki": "wikipedia-cn-20230720-filtered.json"
}

def process_wiki_clean(tokenizer):
    wiki_path = os.path.join(DataPath, FileName['Wiki'])
    with open(wiki_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text = line['completion']
        text_id = tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open('./data/wiki.bin','wb') as f:
        f.write(arr.tobytes())

def DataPreProcess():
    data_path_list=[
        './data/baidubaike_563w_1.bin',
        './data/baidubaike_563w_2.bin',
        './data/baidubaike_563w_3.bin',
        './data/baidubaike_563w_4.bin',
        './data/baidubaike_563w_5.bin',
        './data/medical_book.bin',
        './data/medical_encyclopedia.bin',
        './data/wiki.bin',
        './data/c4_zh_0.bin',
        './data/c4_zh_1.bin',
        './data/c4_zh_2.bin',
        './data/c4_zh_3.bin',
        './data/c4_zh_4.bin',
        './data/c4_zh_5.bin',
        './data/c4_zh_6.bin',
        './data/c4_zh_7.bin',
        './data/c4_zh_8.bin',
        './data/wudaocorpus_zh_0.bin',
        './data/wudaocorpus_zh_1.bin',
        './data/wudaocorpus_zh_2.bin',
        './data/wudaocorpus_zh_3.bin',
        './data/wudaocorpus_zh_4.bin',
        './data/wudaocorpus_zh_5.bin',
        './data/wudaocorpus_zh_6.bin',
        './data/wudaocorpus_zh_7.bin',
        './data/wudaocorpus_zh_8.bin',
        './data/wudaocorpus_zh_9.bin',
        './data/wudaocorpus_zh_10.bin',
        './data/wudaocorpus_zh_11.bin',
        './data/wudaocorpus_zh_12.bin',
        './data/wudaocorpus_zh_13.bin',
        './data/wudaocorpus_zh_14.bin',
        './data/wudaocorpus_zh_15.bin',
        './data/wudaocorpus_zh_16.bin',
    ]

    tokenizer = ChatGLMTokenizer(vocab_file='./Code/Tokenizer/tokenizer.model')

    if (os.path.exists("./data/wiki.bin") == False):
        process_wiki_clean(tokenizer)

    data_lst=[]
    for data_path in tqdm(data_path_list):
        if os.path.exists(data_path):
            with open(data_path,'rb') as f:
                data=np.fromfile(f,dtype=np.uint16)
                data_lst.append(data)
    arr = np.concatenate(data_lst)
    with open('./data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())

    f.close()

    if os.path.exists("./data/pretrain_data.bin") == False:
        print("There is no training data in the datafolder, Please download training data frist.")

    


