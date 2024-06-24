import json

def read_config():
    file_path = './Config/config.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data