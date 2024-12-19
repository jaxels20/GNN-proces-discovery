from src.PetriNet import PetriNet
from src.EventLog import EventLog
import os
from multiprocessing import Pool


class BatchFileLoader:
    """
    A utility class to load PetriNet and EventLog objects in batches or as a whole,
    supporting multiprocessing for efficiency.
    """

    def __init__(self, cpu_count: int = 1):
        """
        Initializes the loader with the number of CPUs to use for multiprocessing.

        Args:
            cpu_count (int): Number of processes to use for multiprocessing.
        """
        self.cpu_count = cpu_count

    @staticmethod
    def _load_petrinet(file_path: str):
        """
        Helper function to load a single PetriNet object from a file.
        Extracts the ID and returns a tuple of (id, PetriNet object).
        Supports both .ptml and .pnml files.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in [".ptml", ".pnml"]:
            return None  # Skip unsupported files

        # Determine how to load the PetriNet based on extension
        if ext == ".ptml":
            pn = PetriNet.from_ptml(file_path)
        else:  # ext == ".pnml"
            pn = PetriNet.from_pnml(file_path)

        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split("_")
        file_id = parts[1] if len(parts) > 1 else name_without_ext

        return file_id, pn

    @staticmethod
    def _load_eventlog(file_path: str):
        """
        Helper function to load a single EventLog object from a file.
        Extracts the ID and returns a tuple of (id, EventLog object).
        """
        if not file_path.endswith(".xes"):
            return None  # Skip non-XES files
        try:
            el = EventLog.load_xes(file_path)
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_")
            file_id = parts[1] if len(parts) > 1 else name_without_ext
        except:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            el = EventLog.load_xes(file_path)
            file_id = name_without_ext

        return file_id, el

    def load_all_petrinets(self, input_dir: str):
        """
        Load all PetriNet objects from a directory using multiprocessing.
        Returns a dictionary where keys are IDs and values are PetriNet objects.
        """
        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                     if f.lower().endswith(".ptml") or f.lower().endswith(".pnml")]
        with Pool(processes=self.cpu_count) as pool:
            results = pool.map(self._load_petrinet, all_files)
        return {file_id: pn for file_id, pn in results if file_id is not None}

    def batch_petrinet_loader(self, input_dir: str, batch_size: int):
        """
        Generator function to yield batches of PetriNet objects from a directory.
        """
        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                     if f.lower().endswith(".ptml") or f.lower().endswith(".pnml")]
        all_files.sort()
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]
            with Pool(processes=self.cpu_count) as pool:
                results = pool.map(self._load_petrinet, batch_files)
            yield {file_id: pn for file_id, pn in results if file_id is not None}

    def load_all_eventlogs(self, input_dir: str):
        """
        Load all EventLog objects from a directory using multiprocessing.
        Returns a dictionary where keys are IDs and values are EventLog objects.
        """
        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".xes")]
        with Pool(processes=self.cpu_count) as pool:
            results = pool.map(self._load_eventlog, all_files)
        
        return {file_id: el for file_id, el in results if file_id is not None}

    def batch_eventlog_loader(self, input_dir: str, batch_size: int):
        """
        Generator function to yield batches of EventLog objects from a directory.
        """
        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".xes")]
        all_files.sort()
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]
            with Pool(processes=self.cpu_count) as pool:
                results = pool.map(self._load_eventlog, batch_files)
            yield {file_id: el for file_id, el in results if file_id is not None}
