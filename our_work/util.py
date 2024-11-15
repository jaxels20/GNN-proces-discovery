from PetriNet import PetriNet
from EventLog import EventLog
import os
from multiprocessing import Pool, cpu_count


def load_petrinet(file_path: str):
    """
    Helper function to load a single PetriNet object from a file.
    Extracts the ID and returns a tuple of (id, PetriNet object).
    """
    if not file_path.endswith(".ptml"):
        return None  # Skip non-PTML files
    pn = PetriNet.from_ptml(file_path)
    # Extract the ID from the filename
    file_id = os.path.basename(file_path).split("_")[1].removesuffix(".ptml")
    return file_id, pn

def load_all_petrinets(input_dir: str, cpu_count=1):
    """
    Load all PetriNet objects from a directory using multiprocessing.
    Returns a dictionary where keys are IDs and values are PetriNet objects.
    """
    # List all files in the input directory
    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".ptml")]

    # Use multiprocessing to load PetriNet objects
    with Pool(processes=cpu_count) as pool:
        results = pool.map(load_petrinet, all_files)

    # Filter out None results (non-PTML files)
    all_petrinets = {file_id: pn for file_id, pn in results if file_id is not None}

    return all_petrinets

def batch_petrinet_loader(input_dir: str, batch_size: int, cpu_count: int = 1):
    """
    Generator function to yield batches of PetriNet objects from a directory.

    Args:
        input_dir (str): Directory containing the PTML files.
        batch_size (int): Number of files to process in each batch.
        cpu_count (int): Number of processes to use for multiprocessing.

    Yields:
        dict: A dictionary where keys are IDs and values are PetriNet objects.
    """
    # List all files in the input directory
    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".ptml")]

    # sort files by name
    all_files.sort()
    
    # Iterate over the files in batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        
        # Use multiprocessing to load PetriNet objects
        with Pool(processes=cpu_count) as pool:
            results = pool.map(load_petrinet, batch_files)
        
        # Filter out None results (non-PTML files)
        batch_petrinets = {file_id: pn for file_id, pn in results if file_id is not None}
        
        yield batch_petrinets

def load_eventlog(file_path: str):
    """
    Helper function to load a single EventLog object from a file.
    Extracts the ID and returns a tuple of (id, EventLog object).
    """
    if not file_path.endswith(".xes"):
        return None  # Skip non-XES files
    try:
        el = EventLog.load_xes(file_path)
        # Extract the ID from the filename
        file_id = os.path.basename(file_path).split("_")[1].removesuffix(".xes")
        if int(file_id) > 1000:  # Quick and dirty way to check if the ID is valid
            raise ValueError("Invalid ID")
    except:
        # If ID extraction or validation fails, use the full filename (without extension) as the ID
        file_id = os.path.basename(file_path).removesuffix(".xes")
    
    return file_id, el

def load_all_eventlogs(input_dir: str, cpu_count=1):
    """
    Load all EventLog objects from a directory using multiprocessing.
    Returns a dictionary where keys are IDs and values are EventLog objects.
    """
    # List all files in the input directory
    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".xes")]

    # Use multiprocessing to load EventLog objects
    with Pool(processes=cpu_count) as pool:
        results = pool.map(load_eventlog, all_files)

    # Filter out None results (non-XES files or failed loads)
    all_eventlogs = {file_id: el for file_id, el in results if file_id is not None}

    return all_eventlogs

def batch_eventlog_loader(input_dir: str, batch_size: int, cpu_count: int = 1):
    """
    Generator function to yield batches of EventLog objects from a directory.

    Args:
        input_dir (str): Directory containing the files.
        batch_size (int): Number of files to process in each batch.
        cpu_count (int): Number of processes to use for multiprocessing.

    Yields:
        dict: A dictionary where keys are IDs and values are EventLog objects.
    """
    # List all files in the input directory
    all_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".xes")]
    # sort the files to ensure consistent ordering
    all_files.sort()

    # Iterate over the files in batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        
        # Use multiprocessing to load EventLog objects
        with Pool(processes=cpu_count) as pool:
            results = pool.map(load_eventlog, batch_files)
        
        # Filter out None results (non-XES files or failed loads)
        batch_eventlogs = {file_id: el for file_id, el in results if file_id is not None}
        
        yield batch_eventlogs