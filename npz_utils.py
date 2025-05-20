import numpy as np
from typing import Dict, Any, Optional
import os

def read_npz_file(file_path: str) -> Dict[str, np.ndarray]:
    """
    Read a .npz file and return its contents as a dictionary of numpy arrays.
    
    Args:
        file_path (str): Path to the .npz file
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing the arrays stored in the .npz file
        where keys are the array names and values are the numpy arrays
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file is not a valid .npz file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if not file_path.endswith('.npz'):
        raise ValueError(f"File {file_path} is not a .npz file")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        # Convert to regular dictionary for easier handling
        return {key: data[key] for key in data.files}
    except Exception as e:
        raise ValueError(f"Error reading NPZ file: {str(e)}")

def get_npz_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about the contents of an NPZ file.
    
    Args:
        file_path (str): Path to the .npz file
        
    Returns:
        Dict[str, Any]: Dictionary containing information about the arrays in the file,
        including their names, shapes, and data types
    """
    arrays = read_npz_file(file_path)
    info = {}
    
    for name, array in arrays.items():
        info[name] = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'size': array.size,
            'memory_usage': array.nbytes
        }
    
    return info

def main():
    """
    Example usage of the NPZ file utilities.
    """
    # Example usage
    try:
        # Replace 'your_file.npz' with your actual NPZ file path
        file_path = 'data/scores_cache/scores_0_1CSE.npz'
        
        # Get information about the NPZ file contents
        info = get_npz_info(file_path)
        print("\nNPZ File Contents:")
        print("-" * 50)
        for name, details in info.items():
            print(f"\nArray name: {name}")
            print(f"Shape: {details['shape']}")
            print(f"Data type: {details['dtype']}")
            print(f"Number of elements: {details['size']}")
            print(f"Memory usage: {details['memory_usage'] / 1024:.2f} KB")
        
        # Read the actual data
        data = read_npz_file(file_path)
        print("\nSuccessfully loaded arrays:", list(data.keys()))
        
        # Print first 10 elements of each array
        print("\nFirst 10 elements of each array:")
        print("-" * 50)
        for name, array in data.items():
            print(f"\nArray: {name}")
            if array.ndim == 1:  # 1D array
                print(array[:20])
            elif array.ndim == 2:  # 2D array
                print(array[:20, :20])  # First 10x10 elements
            else:  # Higher dimensional array
                print(f"Shape: {array.shape}")
                print("First element:", array.flat[0])
                print("Note: Array has more than 2 dimensions, showing only first element")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 