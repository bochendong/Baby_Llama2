import os
import ujson
import pandas as pd 
from ..Utils.Write import write_parquet
from .Punctuation import cleanPunctuation

'''
A Parquet file is a type of columnar storage file format optimized for use with 
large data processing systems like Apache Hadoop and compatible with many data 
analysis tools. It was developed by the Apache Foundation and is widely used in 
data engineering and data science for storing big data.
'''

def lineClean(line, response_threshold=15):
    """
    Processes a single line of input data into a dictionary suitable for a DataFrame.

    Args:
    line (str): A JSON string representing a single record.
    response_threshold (int): Minimum length of the content to keep the record.

    Returns:
    dict or None: A dictionary with the processed data, or None if the data is filtered out.
    """
    item = ujson.loads(line)
    for section in item['sections']:
        response = section['content'].replace('\r', '')
        response = cleanPunctuation(response)
        if len(response) < response_threshold:
            return None
        response_full = section['title'] + section['content']
        return {"response": response_full}


def cleanBaikeData(source_file, target_file, process_func, batch_size=10000):
    """
    Reads data from a source file, clean it, and writes it in batches to a target file.

    Args:
    source_file (str): Path to the source file.
    target_file (str): Path to the target file where processed data will be written.
    process_func (callable): Function to process each line of the source file.
    batch_size (int): Number of records to process before writing to disk.
    """
    raw_line_count, processed_line_count = 0, 0
    current_batch = []

    with open(source_file, 'r', encoding='utf-8') as file:
        for line in file:
            raw_line_count += 1
            processed_data = process_func(line)
            if processed_data is None:
                continue
            processed_line_count += 1
            current_batch.append(processed_data)

            if len(current_batch) >= batch_size:
                df = pd.DataFrame(current_batch)
                write_parquet(target_file, df)
                current_batch = []

    if current_batch:
        df = pd.DataFrame(current_batch)
        write_parquet(target_file, df)

    print(f"Processed {processed_line_count} out of {raw_line_count} lines.")

def cleanBaikeFiles():
    """
    Orchestrates the processing of data files into a parquet format.
    """
    source_files = [
        './data/563w_baidubaike/563w_baidubaike.json',
    ]
    target_file = './data/563w_baidubaike/baike.parquet'
    
    if os.path.exists(target_file) == False:
        for file_name in source_files:
            cleanBaikeData(file_name, target_file, lineClean)
    else:
        print(f"There already exist baike.parquet at {target_file}")

