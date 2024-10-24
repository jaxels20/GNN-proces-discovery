from EventLog import EventLog


if __name__ == "__main__":
    file_name = "data.xes"
    eventlog = EventLog.load_xes(file_name)
    
    # convert the event log to a pm4py log
    pm4py_log = eventlog.to_pm4py()
    
    eventlog_again = EventLog.from_pm4py(pm4py_log)
    
    print(f"Original event log: {eventlog}")
    print(f"Event log from pm4py log: {eventlog_again}")
    
    
    
    
    
    
    
    
        