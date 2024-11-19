from EventLog import EventLog

class TraceListExporter():
    @staticmethod
    def traces_to_xes(traces: list, file_path: str):
        """ Save traces to an XES file """
        # Traces is a list of strings representing activity sequences
        
        eventlog = EventLog.from_trace_list(traces)
        eventlog.to_xes(file_path)
        print(f"Event log saved as {file_path}")