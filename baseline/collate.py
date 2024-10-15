import collections.abc
import re

import torch
from torch.nn import functional as F

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

np_str_obj_array_pattern = re.compile(r"[SaUO]")

def pad_collate(batch, pad_value=0):
    """
    Custom collate function to handle 'dates' separately from tensor data.

    Args:
        batch: List of tuples where each tuple is (data_dict, target).

    Returns:
        A tuple containing:
            - data: Dictionary with batched tensors and a list of dates.
            - targets: Batched target tensors.
    """
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ not in ["str_", "string_"]
    ):
        if elem_type.__name__ in ["ndarray", "memmap"]:
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))
            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    
    elif isinstance(elem, collections.abc.Mapping):
        collated = {}
        for key in elem:
            if key == 'dates':
                # Collect 'dates' as a list of lists without recursion
                collated[key] = [d[key] for d in batch]
            else:
                # Recursively pad other keys
                collated[key] = pad_collate([d[key] for d in batch])
        return collated
    
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    
    elif isinstance(elem, collections.abc.Sequence) and not isinstance(elem, (str, bytes)):
        # Check to make sure that the elements in batch have consistent size
        it = iter(batch)
        try:
            elem_size = len(next(it))
        except StopIteration:
            elem_size = 0
        if not all(len(e) == elem_size for e in it):
            raise RuntimeError("Each element in the batch should be of equal size.")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]
    
    raise TypeError("Format not managed : {}".format(elem_type))
