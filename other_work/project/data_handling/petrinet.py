from project.data_handling.t_component import compute_t_components
from project.data_handling.s_component import compute_s_components
from project.data_handling.drawio import DrawIO

from pm4py.visualization.petri_net import visualizer as vis_factory
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils as utils
from pm4py.evaluation.soundness.woflan.algorithm import apply as woflan
from pm4py.objects.petri import check_soundness

from colorama import Style, Fore
import string
import signal
import itertools
import copy
import random


class PetrinetHandler:
  def __init__(self, fName=''):
    self.mPetrinet = PetriNet(fName)
    self.mInitialMarking = Marking()
    self.mFinalMarking = Marking()
    self.mStochasticInformation = {}

  def get_place_list(self):
    places = []
    for place in self.mPetrinet.places:
      input_transitions = sorted([arc.source.label for arc in place.in_arcs])
      output_transitions = sorted([arc.target.label for arc in place.out_arcs])
      places.append((input_transitions, output_transitions))
    return sorted(places)

  def __hash__(self):
    return hash(str(self.get_place_list()))

  def get_signature(self):
    places = {}
    initials, finals = set(), set()
    for place in self.mPetrinet.places:
      input_transitions = '{' + ','.join(sorted([arc.source.label for arc in place.in_arcs if arc.source.label is not None])) + '}'
      output_transitions = '{' + ','.join(sorted([arc.target.label for arc in place.out_arcs if arc.target.label is not None])) + '}'
      places[place] = f'({input_transitions}->{output_transitions})'
      if place in self.mInitialMarking:
        initials.add(places[place])
      if place in self.mFinalMarking:
        finals.add(places[place])
    silents = []
    for transition in self.mPetrinet.transitions:
      if transition.label is None:
        if len(transition.in_arcs) != 1 or len(transition.out_arcs) != 1:
          raise ValueError('Invalid, since silent transition is not connected to exactly 2 places.')
        in_place = list(transition.in_arcs)[0].source
        out_place = list(transition.out_arcs)[0].target
        silents.append((places[in_place], places[out_place]))
    return set(places.values()), silents, initials, finals

  def equals_other_petrinet(self, other):
    signature1 = self.get_signature()
    signature2 = other.get_signature()

  def copy(self):
    # TODO fix this copying
    copied = PetrinetHandler(self.mPetrinet.name)
    copied.mPetrinet = copy.copy(self.mPetrinet)
    copied.mInitialMarking = copy.copy(self.mInitialMarking)
    copied.mInitialMarking = copy.copy(self.mFinalMarking)
    copied.mStochasticInformation = self.mStochasticInformation
    return copied

  def removeSilentTransitions(self):
    transitions_to_remove = [transition for transition in self.mPetrinet.transitions if transition.label is None]
    for transition in transitions_to_remove:
      utils.remove_transition(self.mPetrinet, transition)

  def importFromFile(self, fFilename):
    fFilename = f'{fFilename}{"" if (fFilename[-5:] == ".pnml") else ".pnml"}'
    elements = pnml_importer.apply(fFilename) #, return_stochastic_information=True)
    if len(elements) == 3:
      self.mPetrinet, self.mInitialMarking, self.mFinalMarking = elements
    elif len(elements) == 4:
      self.mPetrinet, self.mInitialMarking, self.mFinalMarking, self.mStochasticInformation = elements

  def addStartAndEndTransitions(self):
    startTransition = self.addTransition('>')
    for startPlace in self.mInitialMarking:
      self.addArc(startTransition, startPlace)
    endTransition = self.addTransition('|')
    for finalPlace in self.mFinalMarking:
      self.addArc(finalPlace, endTransition)

  def removeStartAndEndTransitions(self):
    self.removeStartTransition()
    self.removeEndTransition()

  def removeStartTransition(self):
    for transition in self.mPetrinet.transitions:
      if transition.label == '>':
        utils.remove_transition(self.mPetrinet, transition)
        break

  def removeEndTransition(self):
    for transition in self.mPetrinet.transitions:
      if transition.label == '|':
        utils.remove_transition(self.mPetrinet, transition)
        break

  def move_initial_marking(self):
    source_place = self.addPlace('source')
    artificial_start_transition = [transition for transition in self.mPetrinet.transitions if transition.label == '>'][0]
    artificial_start_transition.label = None
    self.addArc(source_place, artificial_start_transition)
    self.mInitialMarking = Marking()
    self.mInitialMarking[source_place] += 1

  def move_final_marking(self):
    sink_place = self.addPlace('sink')
    artificial_end_transition = [transition for transition in self.mPetrinet.transitions if transition.label == '|'][0]
    artificial_end_transition.label = None
    self.addArc(artificial_end_transition, sink_place)
    self.mFinalMarking = Marking()
    self.mFinalMarking[sink_place] += 1

  def move_initial_final_markings(self):
    self.move_initial_marking()
    self.move_final_marking()

  def merge_initial_final_marking(self):
    for places_to_merge, marking in zip([list(self.mInitialMarking.keys()), list(self.mFinalMarking.keys())], [self.mInitialMarking, self.mFinalMarking]):
      if len(places_to_merge) > 1:
        for place in places_to_merge[1:]:
          for in_arc in place.in_arcs:
            input_transition = in_arc.source
            if input_transition not in [in_arc.source for in_arc in places_to_merge[0].in_arcs]:
              self.addArc(input_transition, places_to_merge[0])
          for out_arc in place.out_arcs:
            output_transition = out_arc.target
            if output_transition not in [out_arc.target for out_arc in places_to_merge[0].out_arcs]:
              self.addArc(places_to_merge[0], output_transition)
          utils.remove_place(self.mPetrinet, place)
          del marking[place]

  def reduce_silent_transitions(self):
    return

  def splitComplexXORPlaces(self):
    numberOfPlaces = len(self.mPetrinet.places)
    places = list(self.mPetrinet.places)
    placesToRemove = []
    count = 0
    for index in range(numberOfPlaces):
      place = [place for i, place in enumerate(places) if i == index][0]
      if len(place.in_arcs) > 1 and len(place.out_arcs) > 1:
        placesToRemove.append(place)
        initialMarking = place in self.mInitialMarking
        finalMarking = place in self.mFinalMarking
        for in_arc in place.in_arcs:
          count += 1
          newPlace = self.addPlace(f'new_{count}')
          for arc in place.out_arcs:
            self.addArc(newPlace, arc.target)
          if initialMarking:
            self.mInitialMarking[newPlace] += 1
          if finalMarking:
            self.mFinalMarking[newPlace] += 1
          self.addArc(in_arc.source, newPlace)
    for place in placesToRemove:
      utils.remove_place(self.mPetrinet, place)
      for marking in [self.mInitialMarking, self.mFinalMarking]:
        if place in marking:
          del marking[place]

  def silent_transition_exists(self, from_place, to_place):
    for transition in self.mPetrinet.transitions:
      if transition.label is None:
        from_place_ = transition.in_arcs[0].source
        to_place_ = transition.out_arcs[0].target
        if from_place_ == from_place and to_place_ == to_place:
          return True
    return False

  def fromPlaces(self, fPlaces, fTransitionLabels, fPlaceLabels=None, fSilentTransitions=None):
    if fSilentTransitions is None:
      fSilentTransitions  = []

    labels = []
    for index, label in enumerate(fTransitionLabels[:-1]):
      labels.append(label)
      if index == 0:
        labels.append(fTransitionLabels[-1])

    if fPlaceLabels is None:
      fPlaceLabels = [f'p{i}' for i in range(len(fPlaces))]

    places = []
    for place, placeLabel in zip(fPlaces, fPlaceLabels):
      input, output = place.strip('()').split('}, {')
      input = [labels[int(a)] for a in input.strip('{}').split(', ')]
      output = [labels[int(a)] for a in output.strip('{}').split(', ')]
      places.append((place, input, output, placeLabel))

    transitions = {}
    for label in labels:
      transitions[label] = self.addTransition(label)

    silent_transitions_to_add = {}
    added_silents = set()
    for place, inputTransitions, outputTransitions, label in places:
      newPlace = self.addPlace(label)
      for i in range(len(fSilentTransitions)):
        if fSilentTransitions[i][0] == place:
          if i in silent_transitions_to_add.keys():
            if (newPlace, silent_transitions_to_add[i]) not in added_silents:
              silent_transition = self.addTransition(f'st{i}', None)
              self.addArc(newPlace, silent_transition)
              self.addArc(silent_transition, silent_transitions_to_add[i])
              added_silents.add((newPlace, silent_transitions_to_add[i]))
            del silent_transitions_to_add[i]
          else:
            silent_transitions_to_add[i] = newPlace

        if fSilentTransitions[i][1] == place:
          if i in silent_transitions_to_add.keys():
            if (silent_transitions_to_add[i], newPlace) not in added_silents:
              silent_transition = self.addTransition(f'st{i}', None)
              self.addArc(silent_transitions_to_add[i], silent_transition)
              self.addArc(silent_transition, newPlace)
              added_silents.add((silent_transitions_to_add[i], newPlace))
            del silent_transitions_to_add[i]
          else:
            silent_transitions_to_add[i] = newPlace

      for inputTransition in inputTransitions:
        if inputTransition == '>':
          self.mInitialMarking[newPlace] += 1
        self.addArc(transitions[inputTransition], newPlace)
      for outputTransition in outputTransitions:
        if outputTransition == '|':
          self.mFinalMarking[newPlace] += 1
        self.addArc(newPlace, transitions[outputTransition])
    return

  def addPlace(self, fName=''):
    place = PetriNet.Place(fName)
    self.mPetrinet.places.add(place)
    return place

  def addTransition(self, fName='', fLabel=''):
    if fLabel == '':
      fLabel = fName
    transition = PetriNet.Transition(fName, fLabel)
    self.mPetrinet.transitions.add(transition)
    return transition

  def addArc(self, fSource, fTarget, fWeight=1):
    return utils.add_arc_from_to(fSource, fTarget, self.mPetrinet, fWeight)

  def visualize(self, fExport='', fDebug=False):
    parameters = {'debug': True} if fDebug else {}
    gviz = vis_factory.apply(*self.get(), parameters=parameters)
    if fExport != '':
      vis_factory.save(gviz, fExport)
    else:
      vis_factory.view(gviz)

  def get(self, fStochasticInformation=False):
    elements = self.mPetrinet, self.mInitialMarking, self.mFinalMarking
    if fStochasticInformation:
      elements += (self.mStochasticInformation,)

    return elements

  def export(self, fFilename):
    pnml_exporter.apply(self.mPetrinet, self.mInitialMarking, fFilename, final_marking=self.mFinalMarking)

  def get_s_components(self):
    return compute_s_components(self.mPetrinet)

  def get_t_components(self):
    return compute_t_components(self.mPetrinet)

  def simplify_transition_names(self):
    transitions = sorted([t for t in self.mPetrinet.transitions if t.name not in ['>', '|']], key=lambda x: x.label)
    for index, transition in enumerate(transitions):
      transition.name = string.ascii_uppercase[index]
      transition.label = string.ascii_uppercase[index]

  def remove_duplicate_silent_transitions(self):
    transitions = {}
    remove_transitions = []
    for transition in self.mPetrinet.transitions:
      if transition.label is None and len(transition.in_arcs) > 0 and len(transition.out_arcs) > 0:
        from_place, to_place = list(transition.in_arcs)[0].source, list(transition.out_arcs)[0].target
        if from_place == to_place:
          remove_transitions.append(transition)
        else:
          transitions.setdefault((from_place, to_place), []).append(transition)
    [utils.remove_transition(self.mPetrinet, transition) for transition in remove_transitions]
    for id, dup_transitions in transitions.items():
      if len(dup_transitions) > 1:
        [utils.remove_transition(self.mPetrinet, dup_transition) for dup_transition in dup_transitions[1:]]

  def __check_safe_silent_transition(self, silent_transition, keepSafe=1):
    if keepSafe not in [1, 2]:
      print(f'keepSafe argument should be 1 or 2, found {keepSafe}.')
    # Check if silent transition has only one incoming place and one outgoing place
    if keepSafe == 1:
      return len(silent_transition.in_arcs) == 1 and len(silent_transition.out_arcs) == 1

    if len(silent_transition.in_arcs) > 1 or len(silent_transition.out_arcs) > 1:
      return False
    else:
      # Check if both places have already have at least one incoming and at least one outgoing place w/o the st
      in_place = [arc.source for arc in silent_transition.in_arcs][0]
      in_good = len([arc for arc in in_place.out_arcs if arc.target.label is not None]) > 0
      out_place = [arc.target for arc in silent_transition.out_arcs][0]
      out_good = len([arc for arc in out_place.in_arcs if arc.source.label is not None]) > 0

      # Check if one of them is initial of finale marking (not both)
      ini_fin = (int(in_place in self.mInitialMarking or in_place in self.mFinalMarking) +
                 int(out_place in self.mInitialMarking or out_place in self.mFinalMarking)) == 1
      all_good = in_good and out_good
      if not all_good and ini_fin:
        all_good = True

      if all_good:
        return True

    return False

  def label_silent_transitions(self, keepSafe=0):
    if keepSafe == 0:
      silent_transitions = [transition for transition in self.mPetrinet.transitions if transition.label is None]
    else:
      silent_transitions = []
      sts = [transition for transition in self.mPetrinet.transitions if transition.label is None]
      for silent_transition in sts:
        if not self.__check_safe_silent_transition(silent_transition, keepSafe=keepSafe):
          silent_transitions.append(silent_transition)

      if len(silent_transitions) == 0:
        return False

    transition_names = sorted([transition.label for transition in self.mPetrinet.transitions if transition.label is not None])

    if len(transition_names) + len(silent_transitions) <= 26:
      names = string.ascii_lowercase
      current_alphabet_index = names.index(transition_names[-1]) + 1
    else:
      names = [''.join(v) for v in itertools.product(string.ascii_lowercase, string.ascii_lowercase)]
      current_alphabet_index = 0

    for silent_transition in silent_transitions:
      silent_transition.label = names[current_alphabet_index]
      current_alphabet_index += 1

    return True

  def get_places_in_dfs_order(self):
    places = []
    initial_places = sorted([place for place in self.mPetrinet.places if place in self.mInitialMarking or len(place.in_arcs) == 0], key=lambda place: str(place))
    stack = sorted([place for place in self.mPetrinet.places if place in self.mInitialMarking or len(place.in_arcs) == 0], key=lambda place: str(place))

    while len(stack) > 0:
      place = stack.pop()
      if place not in places:
        places.append(place)
        new_places = []
        for new_place in self.get_out_nodes(place):
          if new_place not in places and new_place not in stack:
            new_places.append(new_place)
        stack.extend(sorted(new_places, key=lambda place: str(place)))
    return places, initial_places

  def get_places_in_bfs_order(self):
    places = []
    initial_places = sorted([place for place in self.mPetrinet.places if place in self.mInitialMarking or len(place.in_arcs) == 0], key=lambda place: str(place))
    current_places = sorted([place for place in self.mPetrinet.places if place in self.mInitialMarking or len(place.in_arcs) == 0], key=lambda place: str(place))
    places.extend(current_places)
    while len(places) != len(self.mPetrinet.places):
      new_current_places = []
      for place in current_places:
        for place_out_arc in place.out_arcs:
          for transition_out_arc in place_out_arc.target.out_arcs:
            new_place = transition_out_arc.target
            if new_place not in places and new_place not in new_current_places:
              new_current_places.append(new_place)
      if len(new_current_places) == 0:
        return None, None
      places.extend(sorted(new_current_places, key=lambda p: str(p)))
      current_places = new_current_places
    return places, initial_places

  def get_transitions_in_reverse_bfs_order(self):
    transitions_flat = []
    transitions = []
    current_transitions = [transition for transition in self.mPetrinet.transitions if len(transition.out_arcs) == 0]
    for place in self.mPetrinet.places:
      if place in self.mFinalMarking or len(place.out_arcs) == 0:
        current_transitions.extend([arc.source for arc in place.in_arcs if arc.source not in current_transitions])

    transitions.append(current_transitions)
    transitions_flat.extend(current_transitions)
    while len(transitions_flat) != len(self.mPetrinet.transitions):
      new_current_transitions = []
      for transition in current_transitions:
        for transition_in_arc in transition.in_arcs:
          for place_in_arc in transition_in_arc.source.in_arcs:
            new_transition = place_in_arc.source
            if new_transition not in transitions_flat and new_transition not in new_current_transitions:
              new_current_transitions.append(new_transition)
      if len(new_current_transitions) == 0:
        self.visualize()
        print(a)
      transitions.append(new_current_transitions)
      transitions_flat.extend(new_current_transitions)
      current_transitions = new_current_transitions
    return transitions

  def get_reverse_depth(self, bfs_reversed, transition):
    for index, depth in enumerate(bfs_reversed):
      if transition in depth:
        return index

  def get_transitions_in_bfs_order(self, reverse=False, shuffle_depth=False, depth_in_dfs=True):
    depth = 0
    depth_to_shuffle = 0
    transitions = []
    current_transitions = [transition for transition in self.mPetrinet.transitions if len(transition.in_arcs) == 0]
    for place in self.mPetrinet.places:
      if place in self.mInitialMarking or len(place.in_arcs) == 0:
        current_transitions.extend([arc.target for arc in place.out_arcs if arc.target not in current_transitions])
    current_transitions.sort(key=lambda transition: str(transition), reverse=False)
    if shuffle_depth: # and depth_to_shuffle == depth:
      random.shuffle(current_transitions)
    transitions.extend(current_transitions)
    while len(transitions) != len(self.mPetrinet.transitions):
      depth += 1
      new_current_transitions = []
      for transition in current_transitions:
        for transition_out_arc in transition.out_arcs:
          for place_out_arc in transition_out_arc.target.out_arcs:
            new_transition = place_out_arc.target
            if new_transition not in transitions and new_transition not in new_current_transitions:
              new_current_transitions.append(new_transition)

      new_current_transitions.sort(key=lambda transition: str(transition), reverse=False)
      if shuffle_depth: # and depth_to_shuffle == depth:
        random.shuffle(new_current_transitions)

      transitions.extend(new_current_transitions)
      current_transitions = new_current_transitions
    return transitions

  def get_transitions_inorder(self, order):
    transitions = []
    for name in order:
      transitions.append([t for t in self.mPetrinet.transitions if t.label == name][0])
    return transitions

  def get_transitions_in_dfs_order(self):
    transitions = []
    stack = []
    for place in self.mInitialMarking:
      stack.extend([arc.target for arc in place.out_arcs])
    stack.sort(key=lambda transition: str(transition))

    while len(stack) > 0:
      transition = stack.pop()
      if transition not in transitions:
        transitions.append(transition)
        new_transitions = []
        for new_transition in self.get_out_nodes(transition):
          if new_transition not in transitions:
            new_transitions.append(new_transition)
        stack.extend(sorted(new_transitions, key=lambda transition: str(transition)))
    return transitions

  def get_arcs_in_order(self, sort_places=True, places_sort='bfs', transitions_sort='bfs'):
    if places_sort in ['bfs', 'bfss']:
      places, initial_places = self.get_places_in_bfs_order()
    elif places_sort in ['dfs', 'dfss']:
      places, initial_places = self.get_places_in_dfs_order()
    else:
      return None, None, None, None

    if places_sort in ['bfss', 'dfss']:
      places.shuffle()

    if  places_sort == 'bfs':
      bfs_places, initial_places = places, initial_places
    else:
      bfs_places, initial_places = self.get_places_in_bfs_order()

    if not sort_places:
      bfs_places = sorted(places, key=lambda place: int(str(place).split('\n')[0]))

    if bfs_places is None:
      return None, None, None, None
    place_order = [places.index(place) for place in bfs_places]

    initial_place_indices = []
    if transitions_sort == 'bfs':
      transitions = self.get_transitions_in_bfs_order()
    elif transitions_sort == 'bfsr':
      transitions = self.get_transitions_in_bfs_order(reverse=True)
    elif transitions_sort == 'bfss':
      transitions = self.get_transitions_in_bfs_order(shuffle_depth=True)
    elif transitions_sort == 'dfs':
      transitions = self.get_transitions_in_dfs_order()
    elif transitions_sort == 'dfsr':
      transitions = self.get_transitions_in_dfs_order()
      transitions.reverse()
    elif transitions_sort == 'dfsr':
      transitions = self.get_transitions_in_dfs_order()
      transitions = list(reversed(transitions))
    elif transitions_sort == 'alphabet':
      transitions = sorted([transition for transition in self.mPetrinet.transitions], key=lambda transition: str(transition))
    elif transitions_sort == 'random':
      transitions = self.get_transitions_in_dfs_order()
      random.shuffle(transitions)
    else:
      transitions = [transition for transition in self.mPetrinet.transitions]

    p_t_arcs = []
    t_p_arcs = []
    for i, place in enumerate(places):
      if place in initial_places:
        initial_place_indices.append(i)
      p_t_arcs_ = []
      for arc in place.out_arcs:
        p_t_arcs_.append([i, transitions.index(arc.target)])
      p_t_arcs.extend(sorted(p_t_arcs_, key=lambda x: x[1]))
      t_p_arcs_ = []
      for arc in place.in_arcs:
        t_p_arcs_.append([transitions.index(arc.source), i])
      t_p_arcs.extend(sorted(t_p_arcs_, key=lambda x: x[0]))

    return p_t_arcs, t_p_arcs, place_order, initial_place_indices

  def split_end_places(self):
    places_to_split = []
    for place in self.mPetrinet.places:
      if len(place.out_arcs) == 0 and len(place.in_arcs) > 1:
        places_to_split.append(place)
    i = 0
    for place in sorted(places_to_split, key=lambda place: str(place)):
      for arc in sorted(place.in_arcs, key=lambda arc: str(arc.source)):
        new_place = self.addPlace(f'np{i}')
        i += 1
        self.addArc(arc.source, new_place)
      utils.remove_place(self.mPetrinet, place)

  def get_in_nodes(self, node, two_steps=True):
    nodes = set()
    for in_arc in node.in_arcs:
      if not two_steps:
        nodes.add(in_arc.source)
      else:
        for in_arc2 in in_arc.source.in_arcs:
          nodes.add(in_arc2.source)
    return nodes

  def get_out_nodes(self, node, two_steps=True):
    nodes = set()
    for out_arc in node.out_arcs:
      if not two_steps:
        nodes.add(out_arc.target)
      else:
        for out_arc2 in out_arc.target.out_arcs:
          nodes.add(out_arc2.target)
    return nodes

  def create_unique_end_place(self):
    skip_transitions = []

    added_arcs = []
    for end_place in [p for p in self.mPetrinet.places if len(p.out_arcs) == 0]:
      backwards = set()
      stack = list(self.get_in_nodes(end_place, two_steps=False))
      while len(stack) > 0:
        node = stack.pop()
        if node not in backwards:
          backwards.add(node)
          new_nodes = []
          for new_node in self.get_in_nodes(node, two_steps=False):
            if new_node not in backwards and new_node not in stack:
              new_nodes.append(new_node)
          stack.extend(new_nodes)

      for place in [place for place in backwards if place.__class__.__name__ == 'Place']:
        for place_out_arc in place.out_arcs:
          transition = place_out_arc.target
          if transition not in backwards and transition not in skip_transitions:
            added_arcs.append(self.addArc(transition, end_place))

    extra_transition = self.addTransition('extrae', None)
    for end_place in [p for p in self.mPetrinet.places if len(p.out_arcs) == 0]:
      added_arcs.append(self.addArc(end_place, extra_transition))
    unique_end_place = self.addPlace('sink')
    self.mFinalMarking[unique_end_place] += 1
    added_arcs.append(self.addArc(extra_transition, unique_end_place))
    return unique_end_place

  def create_unique_start_place(self):
    source_place = self.addPlace('source')
    start_transition = self.addTransition('start', None)
    self.addArc(source_place, start_transition)
    for place in list(self.mInitialMarking.keys()):
      self.addArc(start_transition, place)
      del self.mInitialMarking[place]
    self.mInitialMarking[source_place] += 1

  def set_initial_and_final_markings_when_empty(self):
    if len(self.mInitialMarking) == 0:
      source_places = [place for place in self.mPetrinet.places if len([in_arc for in_arc in place.in_arcs if in_arc.source not in [out_arc.target for out_arc in place.out_arcs]]) == 0]
      if len(source_places) == 1:
        self.mInitialMarking[source_places[0]] = 1
      else:
        print('buhh\n')
        [print(place) for place in self.mPetrinet.places]
        self.visualize(fDebug=True)
        raise ValueError('More than one source place.')
    if len(self.mFinalMarking) == 0:
      target_places = [place for place in self.mPetrinet.places if len(place.out_arcs) == 0]
      if len(target_places) == 1:
        self.mFinalMarking[target_places[0]] = 1
      else:
        raise ValueError('More than one target place.')

  def fix_danglings(self):
    new_places = []
    for transition in sorted([transition for transition in self.mPetrinet.transitions if len(transition.in_arcs) == 0], key=lambda x: str(x)):
      utils.remove_transition(self.mPetrinet, transition)
    for transition in sorted([transition for transition in self.mPetrinet.transitions if len(transition.out_arcs) == 0], key=lambda x: str(x)):
      new_places.append(self.addPlace(f'f_{len(new_places)}'))
      self.addArc(transition, new_places[-1])

  def short_circuit(self, unique_start=False, unique_end=True):
    end_places = [self.create_unique_end_place()] if unique_end else [place for place in self.mPetrinet.places if len(place.out_arcs) == 0]
    start_places = [self.create_unique_start_place()] if unique_start else [place for place in self.mInitialMarking]

    short_circuit = self.addTransition('sc')
    for end_place in end_places:
      self.addArc(end_place, short_circuit)
    for start_place in start_places:
      self.addArc(short_circuit, start_place)

    return short_circuit

  def remove_loose_transitions(self):
    transitions_to_remove = [transition for transition in self.mPetrinet.transitions if len(transition.in_arcs) == 0 and len(transition.out_arcs) == 0]
    [utils.remove_transition(self.mPetrinet, transition) for transition in transitions_to_remove]

  def remove_dangling_transitions(self):
    transitions_to_remove = [transition for transition in self.mPetrinet.transitions if len(transition.in_arcs) == 0]
    [utils.remove_transition(self.mPetrinet, transition) for transition in transitions_to_remove]

  def get_s_coverable(self, verbose=True, add_extras=True, remove_extras=True, short_circuit=True):
    new_places = []

    new_transitions = self.short_circuit() if short_circuit else []

    s_components = self.get_s_components()
    s_components_print = [sorted([str(el) for el in comp if el.__class__.__name__ == 'Place']) for comp in s_components]
    elements = set().union(*s_components)
    not_covered_places = self.mPetrinet.places - elements
    if remove_extras:
      [utils.remove_place(self.mPetrinet, place) for place in new_places]
      [utils.remove_transition(self.mPetrinet, transition) for transition in new_transitions]
    if len(not_covered_places) == 0:
      return True, f'{Fore.GREEN}s-coverable {s_components_print}{Style.RESET_ALL}', s_components, not_covered_places
    return False, f'{Fore.RED}not s-coverable {not_covered_places} {s_components_print}{Style.RESET_ALL}', s_components, not_covered_places

  def get_t_coverable(self, initial_place_indices, verbose=True, add_extras=True, remove_extras=True):
    initial_place_names = [f'p{i}' for i in initial_place_indices]
    new_transitions = []
    if add_extras:
      for place in sorted([place for place in self.mPetrinet.places if len(place.in_arcs) == 0 or str(place) in initial_place_names], key=lambda x: str(x)):
        new_transitions.append(self.addTransition(f'temp_{len(new_transitions)}', f'temp_{len(new_transitions)}'))
        self.addArc(new_transitions[-1], place)
      for place in sorted([place for place in self.mPetrinet.places if len(place.out_arcs) == 0], key=lambda x: str(x)):
        new_transitions.append(self.addTransition(f'temp_{len(new_transitions)}', f'temp_{len(new_transitions)}'))
        self.addArc(place, new_transitions[-1])

    t_components = self.get_t_components()
    t_components_print = [sorted([str(el) for el in comp if el.__class__.__name__ == 'Transition']) for comp in t_components]
    elements = set().union(*t_components)
    not_covered_transitions = self.mPetrinet.transitions - elements
    if remove_extras:
      [utils.remove_transition(self.mPetrinet, transition) for transition in new_transitions]

    if len(not_covered_transitions) == 0:
      return True, f'{Fore.GREEN}t-coverable {t_components_print}{Style.RESET_ALL}'
    return False, f'{Fore.RED}not t-coverable {not_covered_transitions} {t_components_print}{Style.RESET_ALL}'

  def get_easy_soundness(self, timeout=None):
    if timeout is None:
      return check_soundness.check_easy_soundness_net_in_fin_marking(self.mPetrinet, self.mInitialMarking, self.mFinalMarking)

    def handler(signum, frame):
      raise Exception('end of time')

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
      easy_soundness = check_soundness.check_easy_soundness_net_in_fin_marking(self.mPetrinet, self.mInitialMarking, self.mFinalMarking)
      signal.alarm(0)
    except Exception:
      print('Time out in checking for easy soundness.')
      easy_soundness = False
    return easy_soundness

  def get_pm4py_soundness(self, timeout=None):
    params = {'return_asap_when_not_sound': True, 'print_diagnostics': True, 'return_diagnostics': True}
    if timeout is None:
      return woflan(self.mPetrinet, self.mInitialMarking, self.mFinalMarking, params)

    def handler(signum, frame):
      raise Exception('end of time')

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
      soundness = woflan(self.mPetrinet, self.mInitialMarking, self.mFinalMarking, params)
      signal.alarm(0)
    except Exception:
      print('Time out')
      soundness = (False, )
    return soundness

  def get_pm4py_s_coverable(self):
    from pm4py.evaluation.soundness.woflan.algorithm import woflan
    from project.data_handling.woflan import step_1_, step_2_, step_3_

    print_diagnostics = False

    woflan_object = woflan(self.mPetrinet, self.mInitialMarking, self.mFinalMarking, print_diagnostics=print_diagnostics)

    short_circuit_transition = self.short_circuit(False, False)
    step_2_res = step_2_(woflan_object, return_asap_when_unsound=True)

    utils.remove_transition(self.mPetrinet, short_circuit_transition)

    if not step_2_res:
      return False, woflan_object.get_output()

    self.create_unique_end_place()
    woflan_object = woflan(self.mPetrinet, self.mInitialMarking, self.mFinalMarking, print_diagnostics=print_diagnostics)

    step_1_res = step_1_(woflan_object, return_asap_when_unsound=True)
    if not step_1_res:
      return False, woflan_object.get_output()
    step_2_res = step_2_(woflan_object, return_asap_when_unsound=True, already_short_circuited=False)
    step_3_res = step_3_(woflan_object, return_asap_when_unsound=True)
    return step_3_res, woflan_object.get_output()

  def get_soundness(self, s_components=None, verbose=True):
    if s_components is None:
      s_components = self.get_s_components()
    s_components_print = [sorted([str(el) for el in comp if el.__class__.__name__ == 'Place']) for comp in
                          s_components]
    elements = set().union(*s_components)
    not_covered_places = self.mPetrinet.places - elements
    if len(not_covered_places) == 0:
      if verbose:
        print(f'{Fore.GREEN}s-coverable {s_components_print}{Style.RESET_ALL}')
      return True
    else:
      new_transitions = []
      for place in sorted(
          [place for place in self.mPetrinet.places if len(place.in_arcs) == 0 or str(place) == 'p0'],
          key=lambda x: str(x)):
        new_transitions.append(self.addTransition(f'temp_{len(new_transitions)}', f'temp_{len(new_transitions)}'))
        self.addArc(new_transitions[-1], place)
      for place in sorted([place for place in self.mPetrinet.places if len(place.out_arcs) == 0],
                          key=lambda x: str(x)):
        new_transitions.append(self.addTransition(f'temp_{len(new_transitions)}', f'temp_{len(new_transitions)}'))
        self.addArc(place, new_transitions[-1])
      t_components = self.get_t_components()

      t_components_print = [sorted([str(el) for el in comp if el.__class__.__name__ == 'Transition']) for comp in
                            t_components]
      elements = set().union(*t_components)
      not_covered_transitions = self.mPetrinet.transitions - elements
      if len(not_covered_transitions) == 0:
        if verbose:
          print(f'{Fore.GREEN}t-coverable {t_components_print}{Style.RESET_ALL}')
        return True
      else:
        print(f'{Fore.RED}not s-coverable {not_covered_places} {s_components_print}{Style.RESET_ALL}')
        print(f'{Fore.RED}not t-coverable {not_covered_transitions} {t_components_print}{Style.RESET_ALL}')
      return False

  def check_joined(self, uncovered_place, covered_places):
    # If there is a way to reach a covered place from uncovered_place, then return True, else False
    if 'f_' in str(uncovered_place):
      return False
    seen_nodes = set()
    stack = [node for node in (self.get_out_nodes(uncovered_place, two_steps=False)) if 'f_' not in str(node)]
    not_reached_f_node = True
    while len(stack) > 0:
      current_node = stack.pop()
      if current_node in covered_places:
        return True
      if current_node not in seen_nodes:
        seen_nodes.add(current_node)
        out_nodes = [out_node for out_node in self.get_out_nodes(current_node, two_steps=False) if out_node not in stack and out_node not in seen_nodes]
        out_nodes_filtered = [out_node for out_node in out_nodes if 'f_' not in str(out_node)]
        if len(out_nodes_filtered) < len(out_nodes):
          not_reached_f_node = False
        stack.extend(out_nodes_filtered)
    return not_reached_f_node

def getPetrinetFromFile(fPetrinetName='', fStochasticInformation=True, fVisualize=False):
  petrinet = PetrinetHandler(fPetrinetName)
  petrinet.importFromFile(fPetrinetName)

  if fVisualize:
    petrinet.visualize()

  return petrinet.get(fStochasticInformation=fStochasticInformation)

def getPetrinetsFromDrawIOFile(filename):
  drawIO = DrawIO(filename)
  petrinet_handlers = []
  for index, model in enumerate(drawIO.models):
    petrinet = PetrinetHandler(str(index))
    places = {place['id']: petrinet.addPlace(place['id']) for place in model['places']}
    for place in model['places']:
      if place['initial_marking']:
        petrinet.mInitialMarking[places[place['id']]] = 1
      if place['final_marking']:
        petrinet.mFinalMarking[places[place['id']]] = 1

    transitions = {transition['id']: petrinet.addTransition(transition['id'], transition['name']) for transition in model['transitions']}
    print(model['transitions'])
    print(model['places'])
    for arc in model['arcs']:
      if arc['from'] is None or arc['to'] is None:
        print(arc)
        continue
      if arc['from'] in places:
        petrinet.addArc(places[arc['from']], transitions[arc['to']])
      else:
        print(arc)
        print(transitions[arc['from']], places[arc['to']])
        petrinet.addArc(transitions[arc['from']], places[arc['to']])
    petrinet_handlers.append(petrinet)
  return petrinet_handlers[0] if len(petrinet_handlers) == 1 else petrinet_handlers


def createSequencePetrinet(fTransitions, fName='', fVisualize=False):
  petrinet = PetrinetHandler(fName)
  currentPlace = petrinet.addPlace('')
  petrinet.mInitialMarking[currentPlace] = 1

  for transition in fTransitions:
    t = petrinet.addTransition(transition)
    petrinet.addArc(currentPlace, t)
    currentPlace = petrinet.addPlace('')
    petrinet.addArc(t, currentPlace)

  petrinet.mFinalMarking[currentPlace] = 1

  if fVisualize:
    petrinet.visualize()

  return petrinet.get()

def getTestPetrinet(fPetrinetName='', fVisualize=False):
  petrinet = PetrinetHandler(fPetrinetName)
  source = petrinet.addPlace('source')
  sink1 = petrinet.addPlace('sink1')
  p1 = petrinet.addPlace('p1')

  a = petrinet.addTransition('a')
  b = petrinet.addTransition('b')
  c = petrinet.addTransition('c')
  d = petrinet.addTransition('d')
  e = petrinet.addTransition('e')

  petrinet.addArc(source, a)
  petrinet.addArc(a, p1)
  petrinet.addArc(p1, c)
  petrinet.addArc(c, p1)
  petrinet.addArc(p1, b)
  petrinet.addArc(b, p1)
  petrinet.addArc(p1, d)
  petrinet.addArc(d, p1)
  petrinet.addArc(p1, e)
  petrinet.addArc(e, sink1)

  petrinet.mInitialMarking[source] = 1
  petrinet.mFinalMarking[sink1] = 1

  if fVisualize:
    petrinet.visualize(fDebug=True)

  print(petrinet.get_s_components())

  return petrinet.get()

def create_petrinet(places, transitions, p_t_arcs, t_p_arcs, initial_place_indices, visualize=False, verbose=False, export='', split=False, remove_danglings=False, short_circuit=False):
  petrinet = PetrinetHandler('')

  places = {name: petrinet.addPlace(f'p{name}') for name in places}
  transitions = {name: petrinet.addTransition(f't{name}') for name in transitions}

  for initial_place_index in initial_place_indices:
    if initial_place_index in places:
      petrinet.mInitialMarking[places[initial_place_index]] += 1

  for pi, ti in p_t_arcs:
    petrinet.addArc(places[pi], transitions[ti])
  for ti, pi in t_p_arcs:
    petrinet.addArc(transitions[ti], places[pi])

  # Create artificial start transition to create a unique source place
  petrinet.create_unique_start_place()

  if remove_danglings:
    petrinet.remove_dangling_transitions()

  if split:
    petrinet.split_end_places()

  fix_danglings = True
  if fix_danglings:
    petrinet.fix_danglings()

  s_coverable, s_report, s_components, not_covered_places = petrinet.get_s_coverable(verbose=verbose, short_circuit=False, remove_extras=True)

  def handler(signum, frame):
    raise Exception('end of time')
  signal.signal(signal.SIGALRM, handler)
  signal.alarm(10)
  try:
    sound, diagnostics = petrinet.get_pm4py_s_coverable()
    signal.alarm(0)
  except Exception:
    print('Time out in checking for s coverability.')
    sound, diagnostics = False, {}

  covered_places = set().union(*s_components)
  if not sound and len(diagnostics.get('uncovered_places_s_component', [])) > 0 and len(covered_places) > 0:
    if s_coverable:
      sound = True
    else:
      uncovered_s_report, covered_s_report = s_report.split('}')
      joined = {}
      for not_covered_place in not_covered_places:
        joined[not_covered_place] = petrinet.check_joined(not_covered_place, covered_places)
        uncovered_s_report = f'{Fore.RED if joined[not_covered_place] else Fore.GREEN}{str(not_covered_place)}'.join(uncovered_s_report.split(str(not_covered_place)))

      if sum(joined.values()) == 0:
        sound = True
      s_report = uncovered_s_report + f'{Fore.GREEN if sound else Fore.RED}' + '}' + covered_s_report

  report = f'{Fore.GREEN if sound else Fore.RED}{sound}{Style.RESET_ALL} - {s_report}'

  if visualize:
    petrinet.visualize(fDebug=True)

  if export != '':
    petrinet.export(export)

  return sound, report

def build_petrinet(p_t_arcs, t_p_arcs, place_order, initial_place_indices, a=2, b=None, visualize=False, verbose=True):
  number_of_places = len(set([arc[0] for arc in p_t_arcs]).union([arc[1] for arc in t_p_arcs]))
  if b is None:
    b = number_of_places + 1
  b = min(b, number_of_places + 1)
  all_sound = True
  reports = []

  for i in range(min(max(a, 2), b - 1), b):
    if verbose:
      print(i)

    p_t_arcs_ = [arc for arc in p_t_arcs if arc[0] in place_order[:i]]
    t_p_arcs_ = [arc for arc in t_p_arcs if arc[1] in place_order[:i]]

    transitions = sorted(list(set([arc[1] for arc in p_t_arcs_]).union(set([arc[0] for arc in t_p_arcs_]))))
    sound, report = create_petrinet(place_order[:i], transitions, p_t_arcs_, t_p_arcs_, initial_place_indices, visualize=(visualize and (i == b - 1)), verbose=verbose, short_circuit=True)

    if report != '':
      reports.append(report)
      if verbose:
        print(report)
    if not sound:
      all_sound = False
  return all_sound, reports
