import re
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

from Utils.Write import write_parquet

NON_CHAR_PATTERN = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")

def generate_minhash(doc, num_perm):
    '''
    Generates a MinHash for a given document.

    Args:
        doc (str): Document text.
        num_perm (int): Number of permutations for MinHash.
        
    Returns:
        MinHash: The generated MinHash object.
    '''
    min_hash = MinHash(num_perm=num_perm)
    for char in doc:
        min_hash.update(char.encode('utf-8'))
    return min_hash

class DuplicateDetector:
    def __init__(self, threshold=0.85, num_perm=256):
        self.index_clusters = defaultdict(set)
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def process_document(self, index, document):
        clean_doc = ''.join(NON_CHAR_PATTERN.split(document))
        doc_hash = generate_minhash(clean_doc, self.num_perm)
        duplicates = self.lsh.query(doc_hash)
        self.lsh.insert(index, doc_hash)
        if duplicates:
            self.index_clusters[min(duplicates)].add(index)

    def get_duplicate_indexes(self):
        all_indexes = set()
        for indexes in self.index_clusters.values():
            all_indexes.update(indexes)
        return all_indexes

def remove_duplicate_rows(input_file, output_file, batch_size=50000):
    detector = DuplicateDetector()
    df = pq.read_table(input_file)
    total_rows = df.num_rows
    print(f"Total rows in dataset: {total_rows}")

    # First pass to detect duplicates
    for index, response in enumerate(df['response']):
        detector.process_document(index, response.as_py())

    # Second pass to filter out duplicates
    to_drop = detector.get_duplicate_indexes()
    kept_rows = [{'response': response.as_py()} for i, response in enumerate(df['response']) if i not in to_drop]

    # Write to Parquet in batches
    for i in range(0, len(kept_rows), batch_size):
        batch_df = pd.DataFrame(kept_rows[i:i+batch_size])
        write_parquet(output_file, batch_df)

def merge_datasets(input_path, output_file, batch_size=50000, max_length=512, min_length=3, truncate_long_texts=False):
    files = get_path_of_suffix_files(input_path, '.parquet')
    all_responses = []

    for file in files:
        print(f"Processing file: {file}")
        df = pq.read_table(file)
        for response in df['response']:
            text = response.as_py()
            if len(text) >= min_length and (len(text) <= max_length or not truncate_long_texts):
                all_responses.append({'response': text[:max_length] if truncate_long_texts else text})

    # Write to Parquet in batches
    for i in range(0, len(all_responses), batch_size):
        batch_df = pd.DataFrame(all_responses[i:i+batch_size])
        write_parquet(output_file, batch_df)

