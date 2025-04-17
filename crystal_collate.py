import torch
from torch_geometric.data import Batch

def crystal_collate(batch):
    """
    Custom collate function to handle batching of PyTorch Geometric Data objects
    along with None values.

    Args:
        batch (list): A list of tuples, where each tuple contains elements returned by __getitem__.

    Returns:
        tuple: A tuple where:
            - Tensor elements are batched (if any),
            - PyTorch Geometric Data objects are batched into a Batch object,
            - None elements are kept as None.
    """
    # Initialize lists to hold batched elements
    batched_none_0 = None  # First element is always None
    batched_data1 = []
    batched_data2 = []
    batched_none_3 = None  # Fourth element is always None
    batched_none_4 = None  # Fifth element is always None

    # Iterate over each sample in the batch
    for sample in batch:
        # Unpack the tuple
        data1 = sample

        # Append Data objects to their respective lists
        batched_data1.append(data1)

        # No action needed for None elements as they are consistently None

    # Batch the Data objects using PyTorch Geometric's Batch.from_data_list
    batched_data1 = Batch.from_data_list(batched_data1)
    batched_data2 = Batch.from_data_list(batched_data2)

    breakpoint()

    # Return the batched tuple
    return (batched_none_0, batched_data1, batched_data2, batched_none_3, batched_none_4)