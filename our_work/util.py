from PetriNet import PetriNet
from EventLog import EventLog
import os

def load_all_petrinets(input_dir: str):
    """ Load all petrinets (process trees)from a directory and return a list of PetriNet objects. 
        Where the keys are filenames and the values are PetriNet objects.
    """
    all_files = os.listdir(input_dir)
    # prepend directory to filenames
    all_files = [os.path.join(input_dir, file) for file in all_files]
    all_petrinets = {} # id -> PetriNet
    for file in all_files:
        if file.endswith(".ptml"):
            pn = PetriNet.from_ptml(file)
            # get the id from the filename
            id = os.path.basename(file).split("_")[1].removesuffix(".ptml")
            all_petrinets[id] = pn
    return all_petrinets

def load_all_eventlogs(input_dir: str):
    """ Load all event logs from a directory and return a dictionary of EventLog objects.
        Where the keys are filenames and the values are EventLog objects.
    """
    all_files = os.listdir(input_dir)
    # prepend directory to filenames
    all_files = [os.path.join(input_dir, file) for file in all_files]
    all_eventlogs = {} # id -> EventLog
    for file in all_files:
        if file.endswith(".xes"):
            el = EventLog.load_xes(file)
            try:
                id = os.path.basename(file).split("_")[1].removesuffix(".xes")
                if int(id) > 1000: # Quick and dirty way to check if the ID is valid
                    raise ValueError("Invalid ID")
            except:
                id = os.path.basename(file).removesuffix(".xes")
                
            all_eventlogs[id] = el
    return all_eventlogs
