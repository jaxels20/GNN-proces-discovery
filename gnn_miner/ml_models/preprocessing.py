from gnn_miner.evaluation.alpha_relations import AlphaRelations
from gnn_miner.data_handling.petrinet import PetrinetHandler
from gnn_miner.data_handling.log import LogHandler

import matplotlib.pyplot as plt
import networkx as nx
import torch as th
import dgl
import itertools
import numpy as np
from collections import defaultdict
import string
import tqdm


class GraphBuilder:
  def __init__(self, fName, fEmbbedingSize, fTraces, fCounts, fTransitions, fLogDirectory='', fDepth=2, fPetrinetHandler=None,
               embedding_strategy='onehot', include_frequency=False, maximum_number_of_places=np.inf):
    self.mName = fName
    self.mLogDirectory = fLogDirectory
    self.mEmbeddingSize = fEmbbedingSize
    self.include_frequency = include_frequency
    self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    self.mTraces = fTraces
    self.mCounts = fCounts
    self.mAlphaRelations = AlphaRelations(fTraces, string.ascii_lowercase)
    self.mNet = dgl.graph(([], []))
    self.mDepth = fDepth
    self.mNumberOfTransitions = len(fTransitions) - 2

    self.mTransitionNames = fTransitions
    if fPetrinetHandler is not None:
      self.mTargetNetWithSilents = self.petrinetToDGLGraph(fPetrinetHandler, fTransitions)
      fPetrinetHandler.removeSilentTransitions()
      self.mTargetNet = self.petrinetToDGLGraph(fPetrinetHandler, fTransitions)
    else:
      self.mTargetNetWithSilents = None
      self.mTargetNet = None

    self.mPossiblePlaces = defaultdict(list)
    self.mPetrinetTransitionLabels = [1, self.mNumberOfTransitions + 2] + list(range(2, self.mNumberOfTransitions + 2))
    self.mPetrinetGraph = self.initializePetrinetGraph(maximum_number_of_places)

    self.mTraceGraph = self.buildTraceGraph()

    if fPetrinetHandler is not None:
      self.mTarget, self.mTargetPlaces = self.getTarget()
      self.place_indices = self.get_place_indices_in_bfs_order()
    else:
      self.mTarget, self.mTargetPlaces = None, None

    self.connectTraceGraphToPetrinet()
    self.addSelfEdges()
    self.mFeatures = self.getFeatures(embedding_strategy)

  def __hash__(self):
    return hash(str(self.mTargetPlaces))

  def getFeatures(self, embedding_strategy):
    if embedding_strategy == 'onehot':
      features = th.zeros((self.mNet.number_of_nodes(), self.mEmbeddingSize), device=self.device)
      for nodeIndex in range(self.mNet.number_of_nodes()):
        nodeLabel = int(self.mNet.nodes[nodeIndex].data['label'].item())
        features[nodeIndex] = onehot(nodeLabel, self.mEmbeddingSize, device=self.device)
    elif embedding_strategy == 'random':
      features = th.zeros((self.mNet.number_of_nodes(), self.mEmbeddingSize), device=self.device)
      for nodeIndex in range(self.mNet.number_of_nodes()):
        nodeLabel = int(self.mNet.nodes[nodeIndex].data['label'].item())
        features[nodeIndex] = onehot(nodeLabel, self.mEmbeddingSize, device=self.device)
    else:
      raise NotImplementedError(f'Embedding strategy ({embedding_strategy}) not implemented, choose one of: [onehot, random].')

    if self.include_frequency:
      features = th.cat((features, self.mNet.nodes[:].data['frequency'].float()), dim=1)

    return features

  def get_place_indices_in_bfs_order(self):
    source = None
    for index, is_initial in enumerate(self.mTargetNetWithSilents.ndata['initial']):
      if is_initial.item():
        source = index
        break
    if source is None:
      raise ValueError('No initial marking found, can\'t perform a directed (dfs/bfs) search for the place order.')

    places_in_order = []
    for step_index, nodes in enumerate(dgl.bfs_nodes_generator(self.mTargetNetWithSilents, source)):
      labels = list(self.mTargetNetWithSilents.nodes[nodes].data['label'].cpu().numpy().flatten())
      node_indices = list(nodes.cpu().numpy())
      places_in_order.extend(sorted([index for index, label in zip(node_indices, labels) if label == 0]))

    silent_transition_label = self.mTransitionNames.index(None) + 1

    place_indices = []
    silent_transition_mapping = {}
    for place in places_in_order:
      incoming_transitions = self.mTargetNetWithSilents.predecessors(place).numpy()
      input_labels = self.mTargetNetWithSilents.nodes[incoming_transitions].data['label']
      input_indices = set([self.mPetrinetTransitionLabels.index(int(l)) for l in input_labels])
      input_silents = set([int(str(l.item()).split('.')[-1]) for l in input_labels if int(l) == silent_transition_label])
      input_indices_ws = input_indices - {silent_transition_label}

      outgoing_transitions = self.mTargetNetWithSilents.successors(place).numpy()
      output_labels = self.mTargetNetWithSilents.nodes[outgoing_transitions].data['label']
      output_indices = set([self.mPetrinetTransitionLabels.index(int(l)) for l in output_labels])
      output_silents = set([int(str(l.item()).split('.')[-1]) for l in output_labels if int(l) == silent_transition_label])
      output_indices_ws = output_indices - {silent_transition_label}

      for index, possiblePlace in enumerate(self.mPossiblePlaces.keys()):
        if input_indices_ws == set(possiblePlace[0]) and output_indices_ws == set(possiblePlace[1]):
          place_indices.append(index)
          silent_transition_mapping[index] = (input_silents, output_silents)
          break

    # TODO Check if all silent transitions occur exactly twice in silent_transition_mapping (once as input, once as output)

    place_and_st_indices = []
    silent_transitions_seen = {}
    number_of_candidate_places = len(self.mPossiblePlaces)
    total_candidates = number_of_candidate_places

    connections = sum([list(range(i)) for i in range(1, len(place_indices) + 1)], [])

    for index, place_index in enumerate(place_indices):
      place_and_st_indices.append(place_index)

      if len(silent_transition_mapping[place_index][0]) == 0 and len(silent_transition_mapping[place_index][1]) == 0:
        continue

      temp_i = sum([i for i in range(index)])

      for silent_transition in silent_transition_mapping[place_index][0]:
        if silent_transition in silent_transitions_seen:
          new_index = connections[temp_i:temp_i + index].index(place_indices.index(silent_transitions_seen[silent_transition]))
          silent_index = number_of_candidate_places + temp_i * 2 + new_index * 2
          place_and_st_indices.append(silent_index)
          del silent_transitions_seen[silent_transition]
        else:
          silent_transitions_seen[silent_transition] = place_index
      for silent_transition in silent_transition_mapping[place_index][1]:
        if silent_transition in silent_transitions_seen:
          new_index = connections[temp_i:temp_i + index].index(place_indices.index(silent_transitions_seen[silent_transition]))
          silent_index = number_of_candidate_places + temp_i * 2 + new_index * 2 + 1
          place_and_st_indices.append(silent_index)
          del silent_transitions_seen[silent_transition]
        else:
          silent_transitions_seen[silent_transition] = place_index

      total_candidates += index * 2

    self.has_silent = max(place_and_st_indices) >= len(self.mPossiblePlaces)

    if len(set(place_and_st_indices)) != len(place_and_st_indices):
      raise ValueError(f'Duplicates found in \'place_and_st_indices\', this should not happen ({place_and_st_indices}).')

    return place_and_st_indices

  def getTarget(self):
    targetVector = [0] * len(self.mPossiblePlaces)
    targetAdjacencyMatrix =  self.mTargetNet.adjacency_matrix(transpose=False).to_dense().numpy()
    targetAdjacencyMatrixT = self.mTargetNet.adjacency_matrix(transpose=True).to_dense().numpy()
    targetPlaces = []
    for index in range(self.mTargetNet.number_of_nodes()):
      if self.mTargetNet.nodes[index].data['label'].item() == 0:
        inputTransitions = set(sorted([self.mPetrinetTransitionLabels.index(int(self.mTargetNet.nodes[j].data['label'].item()))
                                       for j, value in enumerate(targetAdjacencyMatrix[index]) if value == 1]))

        outputTransitions = set(sorted([self.mPetrinetTransitionLabels.index(int(self.mTargetNet.nodes[j].data['label'].item()))
                                        for j, value in enumerate(targetAdjacencyMatrixT[index]) if value == 1]))
        targetPlaces.append(str((sorted(list(inputTransitions)), sorted(list(outputTransitions)))))
        for index2, possiblePlace in enumerate(self.mPossiblePlaces.keys()):
          if inputTransitions == set(possiblePlace[0]) and outputTransitions == set(possiblePlace[1]):
            targetVector[index2] = 1
            break
    return targetVector, sorted(targetPlaces)

  def getTransitions(self):
    distinctTransitions = set()
    for trace in self.mTraces:
      distinctTransitions.update(set(trace))
    return distinctTransitions

  def addEdgeData(self, fGraph, fNumberOfEdges, fReverse=True):
    totalNumberOfEdges = fGraph.number_of_edges()
    if fReverse:
      fGraph.edges[range(totalNumberOfEdges - (fNumberOfEdges * 2), totalNumberOfEdges - fNumberOfEdges)].data['direction'] = th.tensor([[[1], [0]]] * fNumberOfEdges, device=self.device)
      fGraph.edges[range(totalNumberOfEdges - (fNumberOfEdges * 2), totalNumberOfEdges - fNumberOfEdges)].data['frequency'] = th.tensor(np.array([[1.]] * fNumberOfEdges), device=self.device)
      fGraph.edges[range(totalNumberOfEdges - fNumberOfEdges, totalNumberOfEdges)].data['direction'] = th.tensor([[[0], [1]]] * fNumberOfEdges, device=self.device)
      fGraph.edges[range(totalNumberOfEdges - fNumberOfEdges, totalNumberOfEdges)].data['frequency'] = th.tensor(np.array([[1.]] * fNumberOfEdges), device=self.device)
    else:
      fGraph.edges[range(totalNumberOfEdges - fNumberOfEdges, totalNumberOfEdges)].data['direction'] = th.tensor([[[1], [0]]] * fNumberOfEdges, device=self.device)
      fGraph.edges[range(totalNumberOfEdges - fNumberOfEdges, totalNumberOfEdges)].data['frequency'] = th.tensor(np.array([[1.]] * fNumberOfEdges), device=self.device)

  def addPlace(self, fGraph, fInputTransitions, fOutputTransitions, fTraceId=tuple, fVerbose=False):
    if fVerbose:
      print(f'{fInputTransitions} -> {fOutputTransitions}')
    place = (tuple(fInputTransitions), tuple(fOutputTransitions))
    self.mPossiblePlaces[place].append(fTraceId)

    if len(self.mPossiblePlaces[place]) > 1:
      if fVerbose:
        print('Place already added.')
      return

    fGraph.add_nodes(1)
    fGraph.nodes[[fGraph.number_of_nodes() - 1]].data['label'] = th.tensor([[0]], device=self.device)
    fGraph.nodes[[fGraph.number_of_nodes() - 1]].data['trace_id'] = th.tensor([[-1]], device=self.device)
    fGraph.nodes[[fGraph.number_of_nodes() - 1]].data['is_place'] = th.tensor([[1]], device=self.device)

    id = fGraph.number_of_nodes() - 1
    sources = fInputTransitions + [id] * len(fOutputTransitions)
    destinations = [id] * len(fInputTransitions) + fOutputTransitions
    addReverse = True
    if addReverse:
      fGraph.add_edges(sources + destinations, destinations + sources)
    else:
      fGraph.add_edges(sources, destinations)
    self.addEdgeData(fGraph, len(sources), addReverse)


  def getCombinations(self, fTransitions, fMinimumSize=1):
    combinations = []
    for i in range(fMinimumSize, len(fTransitions) + 1):
      combinations.extend(list(itertools.combinations(fTransitions, i)))
    return combinations


  def addXORPlaces(self, fGraph, maximum_number_of_places):
    flatten = lambda l: [item for sublist in l for item in sublist]

    sameInputs  = defaultdict(lambda: defaultdict(set))
    sameOutputs = defaultdict(lambda: defaultdict(set))
    for p, traceIds in self.mPossiblePlaces.items():
      sameInputs[p[0]][p[1]].update(traceIds)
      sameOutputs[p[1]][p[0]].update(traceIds)

    for inputTransitions, outputTransitionGroups in sameInputs.items():
      outputTransitionGroups = [(key, value) for key, value in outputTransitionGroups.items()]
      if len(outputTransitionGroups) > 1:
        outputCombinations = [np.array(combination, dtype=object) for combination in
                              self.getCombinations(outputTransitionGroups, fMinimumSize=2)]
        outputCombinations = [list(combination[:, 0]) for combination in outputCombinations
                              if len(set(flatten(combination[:, 1]))) == len(flatten(combination[:, 1]))]
        for outputTransitions in outputCombinations:
          outputs = list(set(itertools.chain.from_iterable(outputTransitions)))
          is_parallel = False
          for a, b in list(itertools.combinations(outputs, 2)):
            if (a, b) in self.mAlphaRelations.parallel_relations or (b, a) in self.mAlphaRelations.parallel_relations:
              is_parallel = True
              break
          if not is_parallel:
            self.addPlace(fGraph, list(inputTransitions), outputs)
            if len(self.mPossiblePlaces) > maximum_number_of_places:
              break
      if len(self.mPossiblePlaces) > maximum_number_of_places:
        break
    for outputTransitions, inputTransitionGroups in sameOutputs.items():
      inputTransitionGroups = [(key, value) for key, value in inputTransitionGroups.items()]
      if len(inputTransitionGroups) > 1:
        inputCombinations = [np.array(combination, dtype=object) for combination in
                             self.getCombinations(inputTransitionGroups, fMinimumSize=2)]
        inputCombinations = [list(combination[:, 0]) for combination in inputCombinations
                              if len(set(flatten(combination[:, 1]))) == len(flatten(combination[:, 1]))]
        for inputTransitions in inputCombinations:
          inputs = list(set(itertools.chain.from_iterable(inputTransitions)))
          is_parallel = False
          for a, b in list(itertools.combinations(inputs, 2)):
            if (a, b) in self.mAlphaRelations.parallel_relations or (b, a) in self.mAlphaRelations.parallel_relations:
              is_parallel = True
              break
          if not is_parallel:
            self.addPlace(fGraph, inputs, list(outputTransitions))
            if len(self.mPossiblePlaces) > maximum_number_of_places:
              break
      if len(self.mPossiblePlaces) > maximum_number_of_places:
        break


  def addCombinedXORPlaces(self, fGraph):
    sameInputs = defaultdict(set)
    sameOutputs = defaultdict(set)
    for place in self.mPossiblePlaces.keys():
      if len(place[0]) > 1:
        sameInputs[place[0]].add(place[1])
      if len(place[1]) > 1:
        sameOutputs[place[1]].add(place[0])

    for inputTransitions, outputTransitionGroups in sameInputs.items():
      if len(outputTransitionGroups) > 1:
        # outputCombinations = [list(np.array(combination)[:, 0]) for combination in
        #                       self.getCombinations(outputTransitionGroups, fMinimumSize=2)]
        outputCombinations = [
            [sub_combination[0] for sub_combination in combination]
            for combination in self.getCombinations(outputTransitionGroups, fMinimumSize=2)
        ]
        
        for outputTransitions in outputCombinations:
          self.addPlace(fGraph, list(inputTransitions), sorted(outputTransitions))
    
    for outputTransitions, inputTransitionGroups in sameOutputs.items():
      if len(inputTransitionGroups) > 1:
        # inputCombinations = [list(np.array(combination)[:, 0]) for combination in
        #                       self.getCombinations(inputTransitionGroups, fMinimumSize=2)]
        inputCombinations = [
            [sub_combination[0] for sub_combination in combination]
            for combination in self.getCombinations(inputTransitionGroups, fMinimumSize=2)
        ]
        for inputTransitions in inputCombinations:
          self.addPlace(fGraph, sorted(inputTransitions), list(outputTransitions))


  def initializePetrinetGraph(self, maximum_number_of_places=np.inf):
    self.mPetrinetTransitionLabels = [1, self.mNumberOfTransitions + 2] + list(range(2, self.mNumberOfTransitions + 2))
    net = dgl.graph(([], []))
    net.add_nodes(self.mNumberOfTransitions + 2)
    net.nodes[range(self.mNumberOfTransitions + 2)].data['label'] = th.tensor(np.array(self.mPetrinetTransitionLabels).reshape(self.mNumberOfTransitions + 2, 1), device=self.device)
    net.nodes[range(self.mNumberOfTransitions + 2)].data['trace_id'] = th.tensor(np.array([-1] * (self.mNumberOfTransitions + 2)).reshape(self.mNumberOfTransitions + 2, 1), device=self.device)
    net.nodes[range(self.mNumberOfTransitions + 2)].data['frequency'] = th.tensor(np.array([0.] * (self.mNumberOfTransitions + 2)).reshape(self.mNumberOfTransitions + 2, 1), device=self.device)

    for traceIndex, trace in enumerate(self.mTraces):
      trace = [t + 2 for t in trace]
      traceLength = len(trace)
      for depth in range(1, self.mDepth + 1):
        self.addPlace(net, [0], [trace[min(depth - 1, traceLength - 1)]], fTraceId=(traceIndex, 0))
        self.addPlace(net, [trace[traceLength - depth]], [1], fTraceId=(traceIndex, traceLength-depth))
        for i in range(-depth + 1, len(trace) - 1):
          fromTo = (max(i, 0), min(i + depth, traceLength - 1))
          self.addPlace(net, [trace[fromTo[0]]], [trace[fromTo[1]]], fTraceId=(traceIndex, fromTo[0]))
    self.addXORPlaces(net, maximum_number_of_places)
    self.addCombinedXORPlaces(net)

    return net

  def buildTraceGraph(self):
    traceGraph = dgl.graph(([], []))
    traceGraph.add_nodes(2)
    total_count = sum(self.mCounts)
    traceGraph.nodes[[0, 1]].data['label'] = th.tensor([[1], [self.mNumberOfTransitions + 2]], device=self.device)
    traceGraph.nodes[[0, 1]].data['trace_id'] = th.tensor(np.array([[0], [0]]), device=self.device)
    traceGraph.nodes[[0, 1]].data['frequency'] = th.tensor(np.array([[1.], [1.]]), device=self.device)

    for trace_index, (trace, count) in enumerate(zip(self.mTraces, self.mCounts)):
      n = traceGraph.number_of_nodes()
      tn = len(trace)
      traceGraph.add_nodes(tn)
      newNodeIndices = list(range(n, n + tn))
      traceGraph.nodes[newNodeIndices].data['label']    = th.tensor(np.array([t + 2 for t in trace]).reshape(tn, 1), device=self.device)
      traceGraph.nodes[newNodeIndices].data['trace_id'] = th.tensor(np.array([trace_index + 1] * tn).reshape(tn, 1), device=self.device)
      traceGraph.nodes[newNodeIndices].data['frequency'] = th.tensor(np.array(np.array([count / total_count] * tn)).reshape(tn, 1), device=self.device)

      sources = [0] + newNodeIndices
      destinations = newNodeIndices + [1]
      addReverse = True
      if addReverse:
        traceGraph.add_edges(sources + destinations, destinations + sources)
      else:
        traceGraph.add_edges(sources, destinations)
      self.addEdgeData(traceGraph, len(sources), addReverse)
    number_of_nodes = traceGraph.number_of_nodes()
    traceGraph.nodes[:].data['is_place'] = th.tensor(np.array([0] * number_of_nodes).reshape(number_of_nodes, 1), device=self.device)

    return traceGraph

  def petrinetToDGLGraph(self, fPetrinetHandler, fTransitionLabels=string.ascii_lowercase):
    petrinet = fPetrinetHandler.mPetrinet
    net = dgl.graph(([], []))
    net.add_nodes(len(petrinet.transitions) + len(petrinet.places))
    net.nodes[range(net.number_of_nodes())].data['label'] = th.tensor(np.zeros((net.number_of_nodes(), 1)), device=self.device)
    transitions = sorted(list(petrinet.transitions), key=lambda t: t.name)
    labels = []
    silent_counter = 1
    for t in transitions:
      if t.label is not None:
        labels.append(float(fTransitionLabels.index(t.label) + 1))
      else:
        labels.append(float(f'{int(fTransitionLabels.index(t.label) + 1)}.{silent_counter}'))
        silent_counter += 1

    net.nodes[range(len(transitions))].data['label'] = th.tensor(np.array(labels).reshape(len(labels), 1), device=self.device)
    places = list(petrinet.places)

    numberOfTransitions = len(transitions)
    for index, transition in enumerate(transitions):
      net.add_edges([index] * len(transition.out_arcs), [numberOfTransitions + places.index(a.target) for a in transition.out_arcs])
    for index, place in enumerate(places):
      if place in fPetrinetHandler.mInitialMarking:
        net.nodes[numberOfTransitions + index].data['initial'] = th.tensor(1).reshape(1, 1)
      net.add_edges([numberOfTransitions + index] * len(place.out_arcs), [transitions.index(a.target) for a in place.out_arcs])
    return net

  def connectTraceGraphToPetrinet(self):
    n = self.mTraceGraph.number_of_nodes()
    g = dgl.batch([self.mTraceGraph, self.mPetrinetGraph])
    number_of_edges = g.number_of_edges()
    for sourceIndex in range(n):
      destinationIndex = self.mPetrinetTransitionLabels.index(int(g.nodes[sourceIndex].data['label'].item())) + n
      g.add_edges([sourceIndex], [destinationIndex])
      addReverse = False

      USE_FREQUENCY = True
      if USE_FREQUENCY:
        g.edges[number_of_edges + sourceIndex].data['frequency'] = g.nodes[sourceIndex].data['frequency']
      else:
        g.edges[number_of_edges + sourceIndex].data['frequency'] = th.tensor(np.array([[1.]]))

      if addReverse:
        g.add_edges([destinationIndex], [sourceIndex])
      self.addEdgeData(g, 1, addReverse)
    self.mNet = g

  def addSelfEdges(self):
    numberOfNodes = self.mNet.number_of_nodes()
    self.mNet.add_edges(list(range(numberOfNodes)), list(range(numberOfNodes)))
    self.addEdgeData(self.mNet, numberOfNodes, False)

  def visualize(self, fNet='target'):
    nets = {
      'target':   self.mTargetNet,
      'full':     self.mNet,
      'petrinet': self.mPetrinetGraph,
      'trace':    self.mTraceGraph
    }
    net = nets[fNet]
    nx_G = net.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    dcolors = ['w', 'r', 'b', 'g', 'm', 'y', 'k', 'c']
    colors = []
    for i in range(len(net.nodes)):
      node = net.nodes[i]
      colors.append(dcolors[int(node.data['label'].item())])
    nx.draw(nx_G, pos, with_labels=True, node_color=colors)
    plt.show()


def onehot(index, embedding_size, device=None):
  feature_vector = np.zeros((1, embedding_size))
  feature_vector[0][index - 1] = 1
  if device is None:
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
  return th.tensor(feature_vector, device=device)


def get_data(names, fLogPrefix, fPetrinetPrefix='', fEmbeddingSize=13, fDepth=2, fVisualize=False, fExport='', prepare_petrinet=False,
             embedding_strategy='onehot', include_frequency=False, maximum_number_of_places=np.inf, maximum_number_of_transitions=np.inf):
  if isinstance(names, str):
    names = [names]

  max_transitions = 0

  builders = set()
  for i in tqdm.tqdm(names):
    print(i)
    try:
      log_handler = LogHandler(None)
      log_handler._importVariants(f'{fLogPrefix}{i:04d}.npz')
    except FileNotFoundError:
      pass

    transitionLabels = ['>'] + [t[0] for t in log_handler.mTransitions] + [None, '|']
    if len(transitionLabels) > fEmbeddingSize:
      print('-' * 60)
      print(f'Too many transitions: {len(transitionLabels)} > {fEmbeddingSize}')
      continue

    if len(log_handler.mTransitions) > maximum_number_of_transitions:
      print(f'Too many transitions: {len(transitionLabels)} > {maximum_number_of_transitions}')
      continue

    max_transitions = max(max_transitions, len(transitionLabels))

    if fPetrinetPrefix != '':
      petrinet_handler = PetrinetHandler()
      petrinet_handler.importFromFile(f'{fPetrinetPrefix}{i:04d}.pnml')

      if prepare_petrinet:
        petrinet_handler.addStartAndEndTransitions()

      if fVisualize:
        petrinet_handler.visualize()

      if fExport != '':
        petrinet_handler.visualize(fExport=f'{fExport}{i:04d}_true.png')
    else:
      petrinet_handler = None

    percentage = 80
    variants = log_handler.getMostFrequentVariants(percentage, minimum_variants=30, maximum_variants=30)
    traces = [list(variant['variant']) for variant in variants]
    counts = [variant['count'] for variant in variants]
    print(f'{len(log_handler.mVariants)} variants in original log, taking {len(traces)}.')


    pb = GraphBuilder(i, fEmbeddingSize, traces, counts, transitionLabels, fDepth=fDepth, fPetrinetHandler=petrinet_handler, fLogDirectory=fLogPrefix,
                      embedding_strategy=embedding_strategy, include_frequency=include_frequency, maximum_number_of_places=maximum_number_of_places)

    if len(pb.mPossiblePlaces) > maximum_number_of_places:
      print(f'Too many candidate places: {len(pb.mPossiblePlaces)} > {maximum_number_of_places}')
      continue

    if np.sum(pb.mTarget) != len(petrinet_handler.mPetrinet.places):
      print('-' * 60)
      print(f'Not adding this model-log pair (datapoint), since not all places from the target petrinet have been constructed as candidate places. '
            f'Got {np.sum(pb.mTarget)} out of {len(petrinet_handler.mPetrinet.places)}')
    else:
      builders.add(pb)
    print(f'{pb.mNet.number_of_nodes()} nodes, {len(pb.mPossiblePlaces)} candidate places.')

  print(f'Max transitions {max_transitions}.')

  builder_dict = {}
  for builder in builders:
    builder_dict[builder.mName] = builder

  return builder_dict
