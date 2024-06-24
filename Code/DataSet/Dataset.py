import numpy as np
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_paths, max_length=256, use_memmap=False):
        """
        Initializes the dataset.

        Args:
            data_paths (list): List of paths to the data files.
            max_length (int): The maximum length of the sequences.
            use_memmap (bool): Flag to determine if memory mapping should be used to load data.
        """
        super(PretrainDataset, self).__init__()
        self.data = self.load_data(data_paths, max_length, use_memmap)
        print(f"memmap: {use_memmap} train data.shape: {self.data.shape}")
        print("Downloading finished...")

    def load_data(self, data_paths, max_length, use_memmap):
        """
        Load data from files either using memmap or regular numpy arrays.

        Args:
            data_paths (list): List of file paths.
            max_length (int): Maximum sequence length.
            use_memmap (bool): Use numpy memmap to load data.

        Returns:
            np.array: Data array reshaped to (-1, max_length).
        """
        if use_memmap:
            return self.load_data_with_memmap(data_paths[0], max_length)
        else:
            return self.load_data_normally(data_paths, max_length)

    def load_data_with_memmap(self, data_path, max_length):
        """
        Loads data using memory mapping for efficient I/O.

        Args:
            data_path (str): Path to the data file.
            max_length (int): Maximum sequence length.

        Returns:
            np.memmap: Memory-mapped array of data.
        """
        with open(data_path, 'r') as f:
            nbytes = f.seek(0, 2)
            flen = nbytes // np.dtype('uint16').itemsize
        return np.memmap(data_path, dtype=np.dtype('uint16'), mode='r', shape=(flen // max_length, max_length))

    def load_data_normally(self, data_paths, max_length):
        """
        Loads data normally by reading full files into memory.

        Args:
            data_paths (list): List of data files.
            max_length (int): Maximum sequence length.

        Returns:
            np.array: Concatenated data array reshaped to (-1, max_length).
        """
        data_list = []
        for path in data_paths:
            with open(path, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16)
            data_list.append(data)
        data = np.concatenate(data_list)
        data = data[:max_length * (len(data) // max_length)]
        return data.reshape(-1, max_length)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset by index.

        Args:
            index (int): Index of the data item.

        Returns:
            tuple: Tuple of input (X) and target (Y) tensors for training.
        """
        sample = self.data[index]
        X = sample[:-1].astype(np.int64)
        Y = sample[1:].astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)
