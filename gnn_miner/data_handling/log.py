from collections import defaultdict, OrderedDict
import numpy as np
import re

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.importer.xes import importer as xes_import_factory
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.statistics.traces.generic.log import case_statistics

class LogHandler:
  def __init__(self, fLogFilename, fLog=None, fNumberOfTraces=-1):
    self.mLogFilename = fLogFilename
    self.mLog = fLog if fLog is not None else []
    self.mTransitions = OrderedDict()
    if self.mLogFilename is not None:
      self.mLogFilename = fLogFilename if fLogFilename[-4:] == '.xes' else f'{fLogFilename}.xes'
      if fLog is None:
        self.loadLog(fNumberOfTraces)
      else:
        self.__getTransitions()
    self.mVariants = []

  def loadLog(self, fNumberOfTraces):
    parameters = {'max_no_traces_to_import': fNumberOfTraces} if fNumberOfTraces >= 0 else {}
    self.mLog = xes_import_factory.apply(self.mLogFilename, parameters=parameters)
    self.__getTransitions()

  def __getTransitions(self):
    transitions = defaultdict(int)
    for trace in self.mLog:
      for activity in trace:
        activityName = activity['concept:name']
        transitions[activityName] += 1

    self.mTransitions = OrderedDict(sorted(transitions.items(), key=lambda t: t[1], reverse=True))

  def __filterTrace(self, fTransitionList, fTrace, fRegex='', fFrequent=100):
    if fRegex != '':
      fTrace = [activity for activity in fTrace if re.match(fRegex, activity)]
    if fFrequent != 100:
      fTrace = [activity for activity in fTrace if activity in fTransitionList]
    return fTrace

  def __compress(self, fTransitionList, fTrace):
    compressed = [fTransitionList.index(transition) for transition in fTrace]
    return compressed

  def getVariants(self, fRegex='', fFrequent=100):
    if fRegex != '':
      new_transitions = {}
      for transition_name, count in self.mTransitions.items():
        if re.match(fRegex, transition_name):
          new_transitions[transition_name] = count
      print(f'Number of distinct transitions went down from {len(self.mTransitions)} to {len(new_transitions)} due to filtering on regex.')
      self.mTransitions = new_transitions

    fullTransitionList = list(self.mTransitions.keys())
    if fFrequent == 100:
      partialTransitionList = fullTransitionList
    else:
      transitionCount = 0
      totalTransitionCount = sum(self.mTransitions.values())
      partialTransitionList = []
      for index, (transitionName, count) in enumerate(self.mTransitions.items()):
        partialTransitionList.append(transitionName)
        transitionCount += count
        if transitionCount > totalTransitionCount * (fFrequent / 100):
          break
      print(f'Number of distinct transitions went down from {len(fullTransitionList)} to {len(partialTransitionList)} due to filtering on frequency.')

    if len(self.mVariants) == 0:
      variants = case_statistics.get_variant_statistics(self.mLog)
      variants = [{
        'count': int(variant['count']),
        'variant': self.__compress(fullTransitionList,
                                   self.__filterTrace(partialTransitionList, ",".join(variant['variant']).replace(",", ""),
                                                      fRegex=fRegex, fFrequent=fFrequent))
      } for variant in variants]
      if fRegex != '' or fFrequent != 100:
        variantsMerged = defaultdict(int)
        for variant in variants:
          if len(variant['variant']) > 0:
            variantsMerged[tuple(variant['variant'])] += variant['count']
        variants = [{'count': count, 'variant': list(variantTrace)} for variantTrace, count in variantsMerged.items()]
      self.mVariants = sorted(variants, key=lambda x: x['count'], reverse=True)

  def exportLog(self, fTraces, fDirectory, fDatasetName, fTop=10):
    np.save(fDatasetName, np.array([0, 1]))

  def exportVariantsLog(self, fFilename, fTop=-1):
    transitions = np.array([[transition, count] for transition, count in self.mTransitions.items()])

    end = len(self.mVariants) if fTop == -1 else fTop
    variants = np.array([[self.mVariants[0]['count'], np.array(self.mVariants[0]['variant'], dtype=object)]], dtype=object)
    for i in range(1, end):
      a = np.array([self.mVariants[i]['count'], np.array(self.mVariants[i]['variant'], dtype=object)], dtype=object)
      variants = np.append(variants, [a], axis=0)

    np.savez(fFilename, transitions=transitions, variants=variants)

  def _importVariants(self, fFilename):
    if self.mLogFilename is None:
      self.mLogFilename = fFilename
    npz = np.load(f'{fFilename}{"" if fFilename[-4:] == ".npz" else ".npz"}', allow_pickle=True)
    self.mTransitions = OrderedDict()

    for transitionName, count in npz['transitions']:
      self.mTransitions[str(transitionName)] = int(count)

    self.mVariants = []
    for count, variant in npz['variants']:
      self.mVariants.append({'count': int(count), 'variant': variant})

  def getMostFrequentVariants(self, percentage, minimum_variants=np.inf, maximum_variants=np.inf):
    # print('Nr of variants', len(self.mVariants))
    if len(self.mVariants) <= minimum_variants:
      return self.mVariants

    if percentage > 1:
      percentage = percentage / 100
    total_count = sum([variant['count'] for variant in self.mVariants])
    threshold = total_count * percentage
    variants = [self.mVariants[0]]
    current_count = self.mVariants[0]['count']
    index = 1
    while current_count < threshold or len(variants) < minimum_variants:
      variants.append(self.mVariants[index])
      current_count += self.mVariants[index]['count']
      index += 1
      if len(variants) >= maximum_variants:
        break
    print(f'That comes to: {current_count / total_count * 100}%')

    return variants

  def __createTrace(self, fTrace):
    trace = Trace()
    for event in fTrace:
      trace.append(Event({'concept:name': event, 'activity': event}))
    return trace

  def filterOnAttribute(self, fAttributeName, fAttributeValue, fInclude=True):
    newLog = EventLog()
    for trace in self.mLog:
      activities = []
      for activity in trace:
        if fAttributeName in activity and not (activity[fAttributeName] == fAttributeValue) ^ fInclude:
          activities.append(activity['concept:name'])
      if len(activities) > 0:
        newLog.append(self.__createTrace(activities))
    self.mLog = newLog
    # self.__getTransitions()

  def export_to_xes(self, fFilename, fTop=-1):
    # print(self.mTransitions)
    # print(self.mVariants)
    maxxes = []
    for v in self.mVariants:
      maxxes.append(max(v['variant']))
    # print(max(maxxes))
    # print(self.mTransitions)
    # print(len(self.mTransitions))

    if fFilename[-4:] != '.xes':
      fFilename += '.xes'
    xes_log = EventLog()
    transition_names = list(self.mTransitions.keys())
    for variant in self.mVariants:
      trace = [transition_names[event_index] for event_index in variant['variant']]
      for i in range(variant['count']):
        xes_log.append(self.__createTrace(trace))
    xes_exporter.apply(xes_log, fFilename)
