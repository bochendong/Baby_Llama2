import os

# import pandas as pd

from Code.Utils.Split_json import split_json, write_chunks


if __name__ == '__main__':
    input_file = './data/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json'
    chunk_size_mb = 99
    output_dir = './data'
    # Split the JSON file into chunks
    chunks = split_json(input_file, chunk_size_mb)

    # Write each chunk to separate JSON files
    write_chunks(chunks, output_dir)
    '''config = read_config()

    if config["Preprocess"] == True:
        DataPreProcess()
    
    setup_logging("./Log/training.log")
    num_gpus = check_available_gpus()

    mp.set_start_method('spawn')
    processes = []
    for rank in range(num_gpus):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, num_gpus, train, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()'''

