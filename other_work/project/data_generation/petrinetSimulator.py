from copy import copy
import numpy as np
import random
import simpy


class PetrinetSimulator:
  def __init__(self, fPetrinet, fInitialMarking, fFinalMarking, fStochasticInformation):
    self.mPetrinet    = fPetrinet
    self.mPlaces      = fPetrinet.places
    self.mTransitions = fPetrinet.transitions
    self.mStochasticInformation = fStochasticInformation
    self.mInitialMarking = fInitialMarking
    self.mFinalMarking = fFinalMarking

    self.mSimulationEnvironment = None
    self.mTraces = []
    self.mTrace = []
    self.mMaxTraceLength = 0

  def simulateTrace(self, fMaxTraceLength=100, fVerbose=0, fVisualize=False):
    self.__resetSimulation(fMaxTraceLength)
    for place, numberOfTokes in self.mInitialMarking.items():
      for i in range(numberOfTokes):
        self.mSimulationEnvironment.process(self.__simulateStep(place, fVisualize, fVerbose))
    self.mSimulationEnvironment.run(until=fMaxTraceLength+1)
    if len(self.mTrace) == 0:
      return self.mTrace
    self.mTraces.append(self.mTrace)
    return self.mTrace

  def __simulateStep(self, fPlace, fVisualize, fVerbose):
    yield self.mSimulationEnvironment.timeout(0)

    if self.mFinalMarking == self.mCurrentMarking:
      if fVerbose == 1: print('Final marking reached.')
      return

    if len(self.mTrace) >= self.mMaxTraceLength:
      if fVerbose == 1: print('Aborting due to max trace length reached.')
      return

    if len(fPlace.out_arcs) == 0:
      if fVerbose == 1: print('End place reached.')
      return

    # Choose random out arc
    arcs = tuple(fPlace.out_arcs)
    weights = [float(arc.weight) for arc in arcs]
    arc = random.choices(arcs, weights=weights)[0]

    time = np.random.random()

    if arc.target in self.mStochasticInformation.keys():
      time = max(0, self.mStochasticInformation[arc.target].get_value())
    if arc.target.label is None:
      time = 0

    self.mCurrentMarking[fPlace] -= 1
    try:
      arc.mTokens += 1
    except AttributeError:
      arc.mTokens = 1
    self.mSimulationEnvironment.process(self.__simulateTransition(arc.target, time, fVisualize, fVerbose))

  def __simulateTransition(self, fTransition, fTime, fVisualize, fVerbose):
    yield self.mSimulationEnvironment.timeout(fTime)

    for arc in fTransition.in_arcs:
      # TODO remove tokens on arcs, but leave them in places (maybe reserving them).
      if not hasattr(arc, 'mTokens') or arc.mTokens == 0:
        if fVerbose == 1: print(f'Not ready for firing transitions. ({fTransition.name})')
        return

    if fTransition.label is not None and fTransition.label != '|':
      self.mTrace.append((fTransition, self.mSimulationEnvironment.now))

    # Remove tokens from incoming arcs and put in outgoing places
    for arc in fTransition.in_arcs:
      arc.mTokens -= 1
    for arc in fTransition.out_arcs:
      self.mCurrentMarking[arc.target] += 1
      self.mSimulationEnvironment.process(self.__simulateStep(arc.target, fVisualize, fVerbose))

  def __resetSimulation(self, fMaxTraceLength):
    self.mSimulationEnvironment = simpy.Environment()
    self.mMaxTraceLength = fMaxTraceLength
    self.mCurrentMarking = copy(self.mInitialMarking)
    self.mTrace = []
