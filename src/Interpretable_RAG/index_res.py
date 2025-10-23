import faiss

def load_trained_index(index_path):
    """
    Load a pre-trained FAISS index from a file.

    Args:
        index_path (str): Path to the pre-trained index file.

    Returns:
        faiss.Index: The loaded FAISS index.
    """
    print(f"Loading pre-trained index from {index_path}...")
    index = faiss.read_index(index_path)
    print("Index loaded successfully.")
    return index



def save_index(index, output_path):
    """
    Save a FAISS index to a file.

    Args:
        index (faiss.Index): The FAISS index to save.
        output_path (str): Path to save the index file.
    """
    print(f"Saving index to {output_path}...")
    faiss.write_index(index, output_path)
    print("Index saved successfully.")
    



def create_flat_index(d,measure):
    """
    Create a Flat index for exact nearest neighbor search.
    
    Args:
        d (int): Dimensionality of the vectors.
    
    Returns:
        faiss.IndexFlatL2: A FAISS Flat index.
    """
    assert measure.lower() in ['l2','euclidean','inner_product','ip'], "Invalid measure"
    if measure.lower() in ['l2','euclidean']:
        print("Creating a Flat index with L2 distance...")
        index = faiss.IndexFlatL2(d)
        print("Flat index created.")
    elif measure.lower() in ['inner_product','ip']:
        print("Creating a Flat index with Inner Product distance...")
        index = faiss.IndexFlatIP(d)
        print("Flat index created.")
    return index