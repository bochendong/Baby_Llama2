import os
import pandas as pd
import ujson
from ..Utils.Write import write_parquet
from .Punctuation import cleanPunctuation

def cleanWikiEntry(entry, min_length = 15):
    """
    Processes a single Wikipedia entry to format and filter the text.

    Args:
        entry (dict): The Wikipedia entry to process.
        min_length (int): Minimum length of the text to keep the entry.

    Returns:
        dict or None: A dictionary containing the processed text or None if filtered out.
    """
    text = entry.get('completion', '').replace('\r', '')
    text = cleanPunctuation(text)
    if len(text) < min_length:
        return None
    return {'response': text}


def cleanWikiData(read_file, write_to_file, process_func, batch_size =  10000) -> None:
    """
    Reads JSON data from a file, processes it, and writes it to a Parquet file in batches.

    Args:
        read_file (str): Path to the JSON file to read.
        write_to_file (str): Path to the Parquet file to write.
        process_func (callable): Function to process each JSON entry.
        batch_size (int): Number of entries to process before writing to file.
    """
    current_batch = []
    raw_line_cnt, processed_line_cnt = 0, 0
    with open(read_file, 'r', encoding='utf-8') as file:
        data = ujson.load(file)
        for entry in data:
            raw_line_cnt += 1
            processed_entry = process_func(entry)
            if processed_entry is not None:
                processed_line_cnt += 1
                current_batch.append(processed_entry)
            
            if len(current_batch) >= batch_size:
                df = pd.DataFrame(current_batch)
                write_parquet(write_to_file, df)
                current_batch = []

        # Handle any remaining entries after loop
        if current_batch:
            df = pd.DataFrame(current_batch)
            write_parquet(write_to_file, df)

        print(f"Processed {processed_line_cnt} out of {raw_line_cnt} entries.")

def cleanWikiFiles():
    file_names = [
        './data/wikipedia-cn-20230720-filtered.json',
    ]
    target_file = './data/wiki.parquet'
    if os.path.exists(target_file) == False:
        for file_name in file_names:
            read_file = file_name
            cleanWikiData(read_file, target_file, cleanWikiEntry)
    else:
        print(f"There already exist wiki.parquet at {target_file}")
