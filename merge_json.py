import os
import json

def combine_json_chunks(input_dir, output_file):
    combined_data = []

    # Read each chunk file and append to combined_data
    for filename in sorted(os.listdir(input_dir)):
        if filename.startswith('chunk_') and filename.endswith('.json'):
            chunk_file = os.path.join(input_dir, filename)
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                combined_data.extend(chunk_data)
    
    # Write combined data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

# Example usage:
input_dir = './data'
output_file = './data/wikipedia-cn-20230720-filtered.json'

# Combine JSON chunks into the original JSON file
combine_json_chunks(input_dir, output_file)