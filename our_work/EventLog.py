import lxml.etree as ET

class Event:
    """
    Class representing an event in the event log.

    Attributes:
    -----------
    activity : str
        The activity name of the event.
    timestamp : str
        The timestamp of when the event occurred.
    attributes : dict
        Other event attributes (like resource, lifecycle:transition, etc.).
    """
    def __init__(self, activity: str, timestamp: str, attributes: dict):
        self.activity = activity
        self.timestamp = timestamp
        self.attributes = attributes

    def __repr__(self):
        return f"Event(activity={self.activity}, timestamp={self.timestamp}, attributes={self.attributes})"


class Trace:
    """
    Class representing a trace in the event log.

    Attributes:
    -----------
    trace_id : str
        The identifier of the trace.
    events : list[Event]
        A list of events that belong to this trace.
    attributes : dict
        Other trace attributes (like case ID).
    """
    def __init__(self, trace_id: str, attributes: dict):
        self.trace_id = trace_id
        self.events = []
        self.attributes = attributes

    def add_event(self, event: Event):
        """Add an event to the trace."""
        self.events.append(event)

    def __repr__(self):
        return f"Trace(trace_id={self.trace_id}, events={len(self.events)})"


class EventLog:
    """
    Class representing an event log.

    Attributes:
    -----------
    traces : list[Trace]
        A list of traces in the event log.
    """
    def __init__(self):
        self.traces = []

    def load_xes(self, xes_file: str):
        """
        Load an event log from an XES file.

        Parameters:
        -----------
        xes_file : str
            The path to the XES file.
        """
        tree = ET.parse(xes_file)
        root = tree.getroot()

        # Iterate through the traces in the XES file
        for trace in root.findall(".//{*}trace"):
            trace_id = ""
            trace_attributes = {}

            # Parse trace attributes
            for attr in trace.findall("{*}string"):
                if attr.attrib["key"] == "concept:name":
                    trace_id = attr.attrib["value"]
                else:
                    trace_attributes[attr.attrib["key"]] = attr.attrib["value"]

            current_trace = Trace(trace_id, trace_attributes)

            # Iterate through the events in the trace
            for event in trace.findall("{*}event"):
                event_attributes = {}
                activity = ""
                timestamp = ""

                # Parse event attributes
                for attr in event:
                    if attr.attrib["key"] == "concept:name":
                        activity = attr.attrib["value"]
                    elif attr.attrib["key"] == "time:timestamp":
                        timestamp = attr.attrib["value"]
                    else:
                        event_attributes[attr.attrib["key"]] = attr.attrib["value"]

                current_event = Event(activity, timestamp, event_attributes)
                current_trace.add_event(current_event)

            # Add the trace to the log
            self.traces.append(current_trace)

    def __repr__(self):
        """
        Provide a detailed representation of the event log, showing all traces and their events.
        """
        repr_str = f"EventLog with {len(self.traces)} traces:\n"
        for trace in self.traces:
            repr_str += f"Trace ID: {trace.trace_id}, Attributes: {trace.attributes}\n"
            for event in trace.events:
                repr_str += f"  Event: Activity={event.activity}, Timestamp={event.timestamp}, Attributes={event.attributes}\n"
        return repr_str
    
    def get_all_activities(self):
        """
        Get a set of all unique activity names in the event log.
        """
        activities = set()
        for trace in self.traces:
            for event in trace.events:
                activities.add(event.activity)
        return activities

    def get_trace_by_id(self, trace_id: str):
        """
        Retrieve a trace by its trace ID.

        Parameters:
        -----------
        trace_id : str
            The ID of the trace to retrieve.

        Returns:
        --------
        Trace or None
            The trace with the given trace ID, or None if not found.
        """
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None
