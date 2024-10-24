""" Generate toy data for process mining experiments 
    Can generate a list of traces and save them as an XES file
    Usage: python ToyDataGenerator.py --traces "ABCD,ABCBCD,ABCBCBCD" --file "data.xes"
    

"""


import argparse
from EventLog import EventLog

class ToyDataGenerator():
    @staticmethod
    def traces_to_xes(traces: list, file_path: str):
        """ Save traces to an XES file """
        # Traces is a list of strings representing activity sequences
        
        eventlog = EventLog.from_trace_list(traces)
        eventlog.to_xes(file_path)
        print(f"Event log saved as {file_path}")
        

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate an XES file from a list of traces.")
    
    # Add arguments
    parser.add_argument("--traces", type=str, required=True, help="List of traces as a string (e.g., 'ABCD,ABCBCD,ABCBCBCD')")
    parser.add_argument("--file", type=str, required=True, help="Path to save the XES file")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the traces argument into a list
    traces = args.traces.split(",")  # Assuming traces are comma-separated

    # Call the method to generate and save the XES file
    ToyDataGenerator.traces_to_xes(traces, args.file)


        
