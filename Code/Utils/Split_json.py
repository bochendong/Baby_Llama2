import os
import json

def split_json(input_file, chunk_size_mb):
    # Calculate chunk size in bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # Open the original JSON file for reading
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split data into chunks
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for item in data:
        # Convert item to JSON string
        item_str = json.dumps(item, ensure_ascii=False) + '\n'
        item_size = len(item_str.encode('utf-8'))

        # If adding this item exceeds chunk size, start a new chunk
        if current_chunk_size + item_size > chunk_size_bytes:
            chunks.append(current_chunk)
            current_chunk = []
            current_chunk_size = 0
        
        # Add item to current chunk
        current_chunk.append(item)
        current_chunk_size += item_size
    
    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def write_chunks(chunks, output_dir):
    # Write each chunk to a separate file
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f'chunk_{i}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
