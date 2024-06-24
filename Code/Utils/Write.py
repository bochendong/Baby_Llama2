import os
from fastparquet import write

def write_parquet(file_name, data_frame):
    """
    Writes a DataFrame to a Parquet file, appending if the file already exists.

    Args:
    file_name (str): Path to the file where data will be written.
    data_frame (pandas.DataFrame): DataFrame containing the data to write.
    """
    try:
        append = os.path.exists(file_name)
        write(file_name, data_frame, compression='GZIP', append=append)
        print(f"Data successfully written to {file_name}.")
    except Exception as e:
        print(f"Failed to write data to {file_name}: {e}")
