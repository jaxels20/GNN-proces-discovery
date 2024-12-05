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
    
    
def convert_xes_to_npz(xes_file_path, npz_file_path):
    """
    Converts an XES file to an NPZ file that stores transitions and variants.
    :param xes_file_path: Path to the XES file.
    :param npz_file_path: Path to save the NPZ file.
    """
    # Read the XES log
    event_log = pm4py.read_xes(xes_file_path)

    # Count transitions (event names) and collect variants
    transition_counts = OrderedDict()
    variants = []

    for trace in event_log:
        trace_sequence = []
        for event in trace:
            activity = event["concept:name"]  # Get activity name
            trace_sequence.append(activity)
            if activity not in transition_counts:
                transition_counts[activity] = 0
            transition_counts[activity] += 1
        variants.append(tuple(trace_sequence))  # Store as tuple for immutability

    # Prepare transitions array (activity name and count)
    transitions_np = np.array(
        [[activity, count] for activity, count in transition_counts.items()],
        dtype=object
    )

    # Count unique variants and prepare variants array
    variant_counts = OrderedDict()
    for variant in variants:
        if variant not in variant_counts:
            variant_counts[variant] = 0
        variant_counts[variant] += 1

    variants_np = np.array(
        [[count, list(variant)] for variant, count in variant_counts.items()],
        dtype=object
    )

    # Save the transitions and variants into the NPZ file
    np.savez(npz_file_path, transitions=transitions_np, variants=variants_np)

    print(f"Successfully converted {xes_file_path} to {npz_file_path}")     

# Example usage
if __name__ == "__main__":
    save_dir = "./data_dir/evaluation_data/controlled_scenarios"
    xes_file_path = os.path.join(save_dir, "data.xes")
    npz_file_path = os.path.join(save_dir, "data.npz")

    # Example traces (list of strings representing activity sequences)
    traces = ["ABD", "ACD"]
    
    # Create EventLog
    event_log = create_event_log(traces)

    # Save as XES
    save_event_log_as_xes(event_log, xes_file_path)

    # Convert XES to NPZ
    convert_xes_to_npz(xes_file_path, npz_file_path)