import json

def cleanAlpacaGpt4File():
    with open('./data/SFT/alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        q_lst = []
        a_lst = []

        for entry in data:
            q, i, a = entry['instruction'], entry['input'], entry['output']
            q = q + i
            if len(q) < 10 or len(a) < 5:
                continue
            if len(q) > 256 or len(a) > 256:
                continue
            q_lst.append(q)
            a_lst.append(a)

    return q_lst, a_lst

    