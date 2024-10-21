from project.data_generation.petrinetSimulator import PetrinetSimulator

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import PetriNet
# from pm4py.objects.log.exporter.xes import factory as exporter

from tqdm import trange

class Generator:
  def __init__(self, fDatasetName, fPetrinet, fInitialMarking, fFinalMarking, fStochasticInformation):
    self.mDatasetName = fDatasetName
    self.mPetrinet = fPetrinet
    self.mInitialMarking = fInitialMarking
    self.mFinalMarking = fFinalMarking
    self.mStochasticInformation = fStochasticInformation

  def generateData(self, fNumberOfTraces, fDirectory='../git_data/sampleData', fNormalizeWeights=False, fVerbose=False):
    log = EventLog()
    model = PetrinetSimulator(self.mPetrinet, self.mInitialMarking, self.mFinalMarking, self.mStochasticInformation)
    for i in range(fNumberOfTraces):
      trace = model.simulateTrace(fVerbose=0)
      if fVerbose:
        print(trace)
      if len(trace) > 0:
        log.append(self.__createTrace(trace))
    return log

    # outputFilename = f'{fDirectory}{"/" if (fDirectory != "" and fDirectory[-1] != "/") else ""}{self.mDatasetName}.xes'
    # exporter.export_log(log, outputFilename)

  def __createTrace(self, fTrace):
    trace = Trace()
    for event in fTrace:
      if type(event) == PetriNet.Transition:
        trace.append(Event({'concept:name': str(event)}))
      else:
        trace.append(Event({'concept:name': str(event[0]), 'time': event[1]}))
    return trace
