import json

dir_path = '/lustre/orion/bif146/world-shared/enzhi/baby_llama/Baby_Llama2'

def read_config():
    file_path = dir_path + '/Config/config.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data