import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
import numpy as np
from collections import OrderedDict
import os

# Function to count the transitions (events) across traces
def count_transitions(traces):
    transition_counts = OrderedDict()
    for trace in traces:
        for event in trace:
            if event not in transition_counts:
                transition_counts[event] = 0
            transition_counts[event] += 1
    return transition_counts

# Function to convert traces (list of strings) into an EventLog object
def create_event_log(traces):
    event_log = EventLog()
    for trace_str in traces:
        trace = Trace()
        for event_char in trace_str:
            event = Event({DEFAULT_NAME_KEY: event_char})
            trace.append(event)
        event_log.append(trace)
    return event_log

# Function to save event log to an XES file
def save_event_log_as_xes(event_log, file_path):
    pm4py.write_xes(event_log, file_path)

# Function to save event log as an NPZ file (numpy format)
def save_event_log_as_npz(traces, file_path):
    # Count transitions
    transition_counts = count_transitions(traces)

    # Prepare transitions (activities and their occurrence counts)
    transitions = np.array([[event, str(count)] for event, count in transition_counts.items()])

    # Prepare variants (sequences of activity indices)
    unique_traces = set(traces)
    
    # Map each unique event to an index
    event_index = {event: idx for idx, event in enumerate(transition_counts.keys())}

    variants = []
    for trace in unique_traces:
        count = traces.count(trace)
        # Convert the trace string into indices
        variant_indices = np.array([event_index[event] for event in trace])
        variants.append([count, variant_indices])

    # Convert variants to NumPy array (with dtype=object to handle arrays)
    variants_np = np.array(variants, dtype=object)

    # Save transitions and variants into an NPZ file
    np.savez(file_path, transitions=transitions, variants=variants_np)

# Example usage
if __name__ == "__main__":
    # Example traces (list of strings representing activity sequences)
    traces = ["ABCDF", "ABCDF", "ACBEF", "ACBEF"]
    traces = ["ABCD", "ABCBCD", "ABCBCBCD"]
    traces = ["ABC", "ACB"]
    
    # Get the current directory where the script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create EventLog
    event_log = create_event_log(traces)

    # Save as XES
    xes_file_path = os.path.join(current_directory, "data.xes")
    save_event_log_as_xes(event_log, xes_file_path)

    # Save as NPZ (with transitions and variants)
    npz_file_path = os.path.join(current_directory, "data.npz")
    save_event_log_as_npz(traces, npz_file_path)

    print(f"Event log saved as {xes_file_path} and {npz_file_path}")
