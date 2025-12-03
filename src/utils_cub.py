# src/utils_cub.py

import os

def load_cub_attribute_names(cub_root):
    """
    Returns a list attr_names where attr_names[i] is the name of concept c_i.
    """
    attr_file = os.path.join(cub_root, "attributes", "attributes.txt")
    attr_names = []
    with open(attr_file, "r") as f:
        for line in f:
            # example line: "1 bill_shape::curved"
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # attribute_id = int(parts[0])  # 1..K, can ignore
            name = " ".join(parts[1:])
            attr_names.append(name)
    return attr_names
