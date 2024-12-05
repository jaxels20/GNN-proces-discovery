from gnn_miner.ml_models.layers import *

import dgl
from colorama import Fore, Style
import time

# TODO check this.
INCLUDE_FREQUENCY = False

class GATNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dims, out_dim, num_heads, name=''):
    super().__init__()
    self.number_of_layers = len(hidden_dims) + 1
    self.layers = nn.ModuleList([MultiHeadGATLayer(in_dim, hidden_dims[0], num_heads, name=f'gat_{name}_input')])
    for index, (layer_in_dim, layer_out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
      self.layers.append(MultiHeadGATLayer((layer_in_dim + int(INCLUDE_FREQUENCY)) * num_heads, layer_out_dim, num_heads, name=f'gat_{name}_inter_{index}'))
    self.layers.append(MultiHeadGATLayer((hidden_dims[-1] + int(INCLUDE_FREQUENCY)) * num_heads, out_dim, 1, name=f'gat_{name}_output'))

  def prepare_training(self):
    return 0

  def forward(self, graph, hidden_state):
    for index, layer in enumerate(self.layers):
      hidden_state = layer(graph, hidden_state)
      if index < (self.number_of_layers - 1):
        hidden_state = F.relu(hidden_state)
    return hidden_state


class GenerativeModel(nn.Module):
  def __init__(self, embedding_size, include_frequency=False, graph_embedding_type='candidates'):
    super().__init__()
    self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    self.embedding_size = embedding_size

    self.graph_attention_network1 = GATNetwork(in_dim=embedding_size + int(include_frequency), hidden_dims=[32, 64, 32], out_dim=16 - int(INCLUDE_FREQUENCY), num_heads=4, name='1')

    self.graph_attention_network2 = GATNetwork(in_dim=17, hidden_dims=[32], out_dim=16 - int(INCLUDE_FREQUENCY), num_heads=4, name='2')
    self.add_place_agent = AddPlaceAgent(17, graph_embedding_type)
    self.choose_place_agent = ChoosePlaceAgent(16)

  def prepare_for_train(self):
    self.graph_attention_network1.prepare_training()
    self.graph_attention_network2.prepare_training()
    self.add_place_agent.prepare_training()
    self.choose_place_agent.prepare_training()
    self.step_count = 0

  def prepare_for_inference(self):
    self.add_place_agent.prepare_inference()
    self.choose_place_agent.prepare_inference()
    self.step_count = 0

  def get_log_prob(self):
    add_place_log_p    = th.cat(self.add_place_agent.log_probabilities).sum()
    choose_place_log_p = th.cat(self.choose_place_agent.log_probabilities).sum()
    return add_place_log_p + choose_place_log_p

  def get_prob(self):
    add_place_p = th.cat(self.add_place_agent.probabilities).prod()
    choose_place_p = th.tensor(self.choose_place_agent.probabilities).prod()
    return add_place_p * choose_place_p

  def get_score(self, first_connected_index=None):
    # this method doesn't generally prefer short sequences:
    #  taken from https://medium.com/machine-learning-bites/deeplearning-series-sequence-to-sequence-architectures-4c4ca89e5654
    alpha = 0.7
    if first_connected_index is not None:
      ps = th.tensor(self.choose_place_agent.probabilities)
      max_score, max_index = th.tensor([th.log(ps[:i]).sum() / (i**alpha) for i in range(first_connected_index, len(ps) + 1)]).max(0)
      return max_score, first_connected_index + max_index.item()
    else:
      all_probs = th.cat((th.cat(self.add_place_agent.probabilities).flatten(), th.tensor(self.choose_place_agent.probabilities)))
      score = th.log(all_probs).sum() / len(all_probs)**alpha
      return score, None

  @property
  def action_step(self):
    old_step_count = self.step_count
    self.step_count += 1
    return old_step_count

  def forward_train(self, graph, features, number_of_places, alpha_relations, places):
    self.prepare_for_train()

    original_graph_size = graph.number_of_nodes()
    number_of_non_candidates = original_graph_size - number_of_places
    decisions = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    decisions[-number_of_places:] = -1.
    graph.ndata['decision'] = th.tensor(decisions, device=self.device)
    graph.ndata['silent_np'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)
    graph.ndata['is_silent'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)

    end_transition_label = int(max(graph.ndata['label']).numpy())

    h = self.graph_attention_network1(graph, features)
    h = th.cat((h, graph.nodes[:].data['decision'].reshape(original_graph_size, 1)), dim=1)

    while self.add_place_agent(graph, h, int(len(places) != self.step_count))[0]:
      h = self.graph_attention_network2(graph, h)
      _ = self.choose_place_agent(graph, h, place=places[self.action_step], number_of_candidates=(graph.number_of_nodes() - number_of_non_candidates))

      # Create all possible silent transitions between all already chosen places and the new places. These can also be chosen as a 'candidate place' in a next round.
      if places[self.step_count - 1] < number_of_places:
        h = self._create_silent_transitions(graph, h, places[:self.step_count], number_of_places, (graph.number_of_nodes() - number_of_non_candidates), alpha_relations, end_transition_label)
      h = th.cat((h, graph.nodes[:].data['decision'].reshape(graph.number_of_nodes(), 1)), dim=1)

    return -self.get_log_prob()

  def forward_initialize(self, graph, features, number_of_places):
    self.prepare_for_inference()
    original_graph_size = graph.number_of_nodes()
    decisions = np.zeros(original_graph_size, dtype=np.float32)
    decisions[-number_of_places:] = -1.
    graph.ndata['decision'] = th.tensor(decisions, device=self.device)
    graph.ndata['unsound'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)
    graph.ndata['silent_np'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)
    graph.ndata['is_silent'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)

    h = self.graph_attention_network1(graph, features)
    h = th.cat((h, graph.nodes[:].data['decision'].reshape(original_graph_size, 1)), dim=1)
    return h

  def forward_step(self, graph, h, number_of_non_candidates):
    # TODO MAYBE: set unsound value to zero for everything again. I'm not sure if a place could be valid with an extended subgraph. Not doing this could speed the process up quite a bit.
    graph.ndata['unsound'] = th.tensor(np.zeros(graph.number_of_nodes(), dtype=np.int64), device=self.device)

    # Propagation
    h = self.graph_attention_network2(graph, h)
    n = graph.number_of_nodes()
    number_of_candidates = n - number_of_non_candidates

    place_probs = self.choose_place_agent(graph, h, place=None, number_of_candidates=number_of_candidates, top=0, return_probs=True)
    sorted_probs = th.argsort(place_probs, descending=True)
    return sorted_probs, place_probs, h

  def forward_inference(self, graph, features, number_of_places, alpha_relations, route=None, prune=-np.inf, max_number_of_places=np.inf, length_normalization=False, check_s_coverability_callback=lambda x, y: True):
    self.prepare_for_inference()
    original_graph_size = graph.number_of_nodes()
    number_of_non_candidates = original_graph_size - number_of_places
    decisions = np.zeros(original_graph_size, dtype=np.float32)
    decisions[-number_of_places:] = -1.
    graph.ndata['decision']  = th.tensor(decisions, device=self.device)
    graph.ndata['unsound']   = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)
    graph.ndata['silent_np'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)
    graph.ndata['is_silent'] = th.tensor(np.zeros(original_graph_size, dtype=np.int64), device=self.device)

    end_transition_label = int(max(graph.ndata['label']).numpy())

    h = self.graph_attention_network1(graph, features)
    h = th.cat((h, graph.nodes[:].data['decision'].reshape(original_graph_size, 1)), dim=1)

    pruned_route = None

    choice_times = []
    current_score = th.tensor([0.])
    places = []
    connectedIndex = None
    max_number_of_places = min(number_of_places, max_number_of_places)
    connected = self._check_connectedness(graph, places)
    unsounds = set()
    while len(places) < max_number_of_places:
      add_place, probability = self.add_place_agent(graph, h, None)
      # TODO improve logging.
      print('addplace', add_place, probability)
      if not add_place and connected:
        break
      print(f'Number of places: {len(places)}')
      # TODO MAYBE: set unsound value to zero for everything again. I'm not sure if a place could be valid with an extended subgraph. Not doing this could speed the process up quite a bit.
      graph.ndata['unsound'] = th.tensor(np.zeros(graph.number_of_nodes(), dtype=np.int64), device=self.device)

      h = self.graph_attention_network2(graph, h)
      n = graph.number_of_nodes()
      number_of_candidates = n - number_of_non_candidates
      none_found = False
      number_candidates_tried = 0
      if route is not None:
        # TODO add counter for when all candidates have been tried and none can make the net s-coverable.
        top = route[self.action_step]
        place_probs = self.choose_place_agent(graph, h, place=None, number_of_candidates=number_of_candidates, top=top, return_probs=True)

        sorted_probs = th.argsort(place_probs, descending=True)
        print(sorted_probs)
        top_temp = 0
        for place in sorted_probs:
          place = place.item()
          if place in unsounds:
            continue
          s_coverable, description = check_s_coverability_callback(graph, places + [place])
          if s_coverable and top_temp == top:
            break
          elif s_coverable:
            top_temp += 1
          elif not s_coverable and description == 'ns':
            unsounds.add(place)
        else:
          none_found = True

        if not none_found:
          # Found the appropriate place, now set the probability and the choice
          print(Fore.GREEN, 'prob:', place_probs[place].item(), Style.RESET_ALL)
          if place_probs[place].item() == 0:
            return places, th.tensor([-1.e10000]), None, self.choose_place_agent.probabilities, self.add_place_agent.add_probabilities
          self.choose_place_agent.probabilities.append(place_probs[place].data)
          graph.nodes[number_of_non_candidates + place].data['decision'] = th.tensor([1.], device=self.device)
          places.append(place)
      else:
        place = self.choose_place_agent(graph, h, place=None, number_of_candidates=number_of_candidates)
        s_coverable = check_s_coverability_callback(graph, places + [place])
        while not s_coverable:
          # TODO REVERT (self.probabilities.append(place_probs[0][place].data), graph.nodes[n - number_of_candidates + place].data['decision'] = th.tensor([1.], device=self.device))
          self.choose_place_agent.probabilities.pop()
          graph.nodes[n - number_of_candidates + place].data['decision'] = th.tensor([0.], device=self.device)
          # TODO set the place as such, such that it won't be chosen again
          graph.nodes[n - number_of_candidates + place].data['unsound'] = th.tensor([1], device=self.device)
          place = self.choose_place_agent(graph, h, place=None, number_of_candidates=number_of_candidates)
          s_coverable = check_s_coverability_callback(graph, places + [place])
          number_candidates_tried += 1
          if number_candidates_tried == min(number_of_candidates, max_number_of_places):
            none_found = True
            break
        places.append(place)

      add_place_probabilities = self.add_place_agent.add_probabilities
      place_choice_probabilities = self.choose_place_agent.probabilities
      print(none_found)
      if none_found:
        print(f'None found, tried {min(number_of_candidates, max_number_of_places)} candidates but all result in a non s-coverable petri net.')
        return places, th.tensor([-1.e10000]), None, place_choice_probabilities, add_place_probabilities

      if length_normalization:
        current_score, _ = self.get_score(len(places))
      else:
        current_score += np.log(max(th.tensor([1.e-10]), self.add_place_agent.add_probabilities[-1].data.item()))
        current_score += np.log(max(th.tensor([1.e-10]), self.choose_place_agent.probabilities[-1].data.item()))
        if current_score < prune:
          pruned_route = route[:self.step_count]
          break

      if places[-1] < number_of_places:
        h = self._create_silent_transitions(graph, h, places, number_of_places, (graph.number_of_nodes() - number_of_non_candidates), alpha_relations, end_transition_label)
      h = th.cat((h, graph.nodes[:].data['decision'].reshape(graph.number_of_nodes(), 1)), dim=1)

      connected = self._check_connectedness(graph, places)
      if connected and connectedIndex is None:
        connectedIndex = len(places)

      choice_time = time.time() - prop_time
      choice_times.append(choice_time)
      prop_time = time.time()

    # TODO improve logging.
    print('addplace2', add_place)

    if length_normalization:
      score, cutoff_index = self.get_score(connectedIndex)
      if cutoff_index is not None:
        places = places[:cutoff_index]
    else:
      score = current_score

    return places, score, pruned_route, place_choice_probabilities, add_place_probabilities

  def forward(self, graph, features, number_of_places, alpha_relations, places=None, route=None, prune=0, max_number_of_places=np.inf, length_normalization=False, check_s_coverability_callback=lambda x, y: (True, '')):
    if self.training:
      return self.forward_train(graph, features, number_of_places, alpha_relations, places=places)
    else:
      return self.forward_inference(graph, features, number_of_places, alpha_relations, route=route, prune=prune, max_number_of_places=max_number_of_places, length_normalization=length_normalization, check_s_coverability_callback=check_s_coverability_callback)

  def _check_connectedness(self, graph, places):
    # Search for unconnected transitions and return those
    first_transition_index = max(np.nonzero(graph.ndata['label'].numpy().flatten() == 1)[0])
    first_place_index = min(np.nonzero(graph.ndata['label'].numpy().flatten() == 0)[0])
    transition_indices = [transition_index for transition_index in range(first_transition_index, first_place_index) if \
                          min(graph.predecessors(transition_index)) < first_transition_index]
    discovered_places = set([place + first_place_index for place in places])

    in_unconnected_transitions, out_unconnected_transitions = set(), set()
    for index, petrinet_transition_index in enumerate(transition_indices):
      input_places, output_places = self._get_neighbors(graph, petrinet_transition_index, min_neighbor_index=first_place_index)
      if len(input_places & discovered_places) == 0 and index != 0 and len(input_places) > 0:
        in_unconnected_transitions.add(petrinet_transition_index)
      if len(output_places & discovered_places) == 0 and index != 1 and len(output_places) > 0:
        out_unconnected_transitions.add(petrinet_transition_index)

    if len(in_unconnected_transitions) > 0 or len(out_unconnected_transitions) > 0:
      return False, in_unconnected_transitions, out_unconnected_transitions

    graph_node_indices = transition_indices + list(discovered_places)
    if len(graph_node_indices) > sum([len(nds) for nds in dgl.bfs_nodes_generator(graph.subgraph(graph_node_indices), 0)]):
      return False, set(), set()

    return True, set(), set()

  def clean_graph(self, graph, original_graph_size):
    for key in list(graph.ndata.keys()):
      graph.nodes[:].data[key] = graph.nodes[:].data[key].detach()
    for key in list(graph.edata.keys()):
      graph.edges[:].data[key] = graph.edges[:].data[key].detach()

    nodes_to_remove = list(range(original_graph_size, graph.number_of_nodes()))
    graph.remove_nodes(nodes_to_remove)

  def get_alpha_label(self, label, max_label):
    conversion = {1: 0, max_label: 1}  # start/end transition is labeled 1/max in graph and 0/1 in alpha relations
    return conversion.get(label, label)

  def _check_directly_follows(self, graph, incoming_transitions, outgoing_transitions, inside_transitions, alpha_relations, end_transition_label, verbose=False):
    # TODO check code, possibly contains a bug, but not sure how to reproduce it.
    parallel_transition_labels = set()
    for inside_transition in inside_transitions:
      inside_transition_label = self.get_alpha_label(int(graph.nodes[inside_transition].data['label'][0][0].numpy()), end_transition_label)
      parallel_transition_labels.update([t[1] for t in alpha_relations.parallel_relations if t[0] == inside_transition_label])

    directly_follows = False
    for incoming_transition in incoming_transitions:
      if graph.nodes[incoming_transition].data['is_silent'] == 1 or graph.nodes[incoming_transition].data['is_place'] == 1:
        continue

      incoming_transition_label = self.get_alpha_label(int(graph.nodes[incoming_transition].data['label'][0][0].numpy()), end_transition_label)

      for outgoing_transition in outgoing_transitions:
        if graph.nodes[outgoing_transition].data['is_silent'].item() == 1 or graph.nodes[outgoing_transition].data['is_place'].item() == 1:
          continue

        outgoing_transition_label = self.get_alpha_label(int(graph.nodes[outgoing_transition].data['label'][0][0].numpy()), end_transition_label)

        if outgoing_transition_label in alpha_relations.directly_follows_relations_dict[incoming_transition_label] and outgoing_transition_label not in parallel_transition_labels:
          directly_follows = True
          break

        if outgoing_transition_label not in alpha_relations.directly_follows_relations_dict[incoming_transition_label]:
          for parallel_transition_label in parallel_transition_labels:
            if parallel_transition_label in alpha_relations.directly_follows_relations_dict[incoming_transition_label]:
              break
          else:
            directly_follows = False
            break
      if not directly_follows:
        break
    return directly_follows


  def _get_neighbors(self, graph, node_id, min_neighbor_index=0):
    neighbors = graph.successors(node_id).numpy()
    neighbors = neighbors[neighbors >= min_neighbor_index]
    newest_place_incoming_transitions = set()
    newest_place_outgoing_transitions = set()
    for neighbor in neighbors:
      edges = graph.edge_ids(node_id, neighbor, return_uv=True)[2]
      for edge_direction in graph.edata['direction'][edges]:
        if edge_direction[0][0] == 1:
          newest_place_outgoing_transitions.add(neighbor)
        else:
          newest_place_incoming_transitions.add(neighbor)
    return newest_place_incoming_transitions, newest_place_outgoing_transitions

  def add_silent_transition(self, graph, id, from_place, to_place):
    graph.add_nodes(1)
    graph.nodes[id].data['decision'] = th.tensor([-1.], device=self.device)
    graph.nodes[id].data['silent_np'] = th.tensor([-1], device=self.device)
    graph.nodes[id].data['is_silent'] = th.tensor([1], device=self.device)
    number_of_edges = graph.number_of_edges()
    graph.add_edges([from_place, id, id], [id, to_place, id])
    graph.edges[range(number_of_edges, number_of_edges + 3)].data['direction'] = th.tensor([[[1], [0]]] * 3, device=self.device)
    graph.add_edges([id, to_place], [from_place, id])
    graph.edges[range(number_of_edges + 3, number_of_edges + 5)].data['direction'] = th.tensor([[[0], [1]]] * 2, device=self.device)
    graph.edges[range(number_of_edges, number_of_edges + 5)].data['frequency'] = th.tensor(np.array([[1.]] * 5), device=self.device)
    return

  def _create_silent_transitions(self, graph, hidden_states, places, number_of_places, number_of_choices, alpha_relations, end_transition_label):
    number_of_nodes_before = graph.number_of_nodes()
    place_start_index = number_of_nodes_before - number_of_choices
    old_places = [place_start_index + p for p in places[:-1] if p < number_of_places]
    nr_of_new_silent_transitions = 2 * len(old_places)

    if nr_of_new_silent_transitions == 0:
      return hidden_states

    graph.add_nodes(nr_of_new_silent_transitions)
    newest_place = place_start_index + places[-1]
    newest_place_incoming_transitions, newest_place_outgoing_transitions = self._get_neighbors(graph, newest_place)

    new_silent_transition_indices = list(range(number_of_nodes_before, number_of_nodes_before + nr_of_new_silent_transitions))
    graph.nodes[new_silent_transition_indices].data['decision']  = th.tensor([-1.] * nr_of_new_silent_transitions, device=self.device)
    graph.nodes[new_silent_transition_indices].data['silent_np'] = th.tensor([-1] * nr_of_new_silent_transitions, device=self.device)
    graph.nodes[new_silent_transition_indices].data['is_silent'] = th.tensor([1] * nr_of_new_silent_transitions, device=self.device)

    for index, old_place in enumerate(old_places):

      old_place_incoming_transitions, old_place_outgoing_transitions = self._get_neighbors(graph, old_place)

      forward_silent_transition_index  = new_silent_transition_indices[index * 2]
      backward_silent_transition_index = new_silent_transition_indices[index * 2 + 1]
      if not self._check_directly_follows(graph, old_place_incoming_transitions, newest_place_outgoing_transitions, {*newest_place_incoming_transitions, *old_place_outgoing_transitions}, alpha_relations, end_transition_label):
        graph.nodes[forward_silent_transition_index].data['silent_np'] = th.tensor([1], device=self.device)

      if not self._check_directly_follows(graph, newest_place_incoming_transitions, old_place_outgoing_transitions, set(), alpha_relations, end_transition_label):
        graph.nodes[backward_silent_transition_index].data['silent_np'] = th.tensor([1], device=self.device)

      # Set initial states
      new_hidden_state = ((hidden_states[old_place] + hidden_states[newest_place]) / 2).reshape(1, len(hidden_states[old_place]))
      hidden_states = th.cat((hidden_states, new_hidden_state, new_hidden_state))

      number_of_edges = graph.number_of_edges()
      # Forward edges
      graph.add_edges([old_place, forward_silent_transition_index, newest_place, backward_silent_transition_index, forward_silent_transition_index, backward_silent_transition_index],
                      [forward_silent_transition_index, newest_place, backward_silent_transition_index, old_place, forward_silent_transition_index, backward_silent_transition_index])
      graph.edges[range(number_of_edges, number_of_edges + 6)].data['direction'] = th.tensor([[[1], [0]]] * 6, device=self.device)

      # Backward edges
      graph.add_edges([forward_silent_transition_index, newest_place, backward_silent_transition_index, old_place],
                      [old_place, forward_silent_transition_index, newest_place, backward_silent_transition_index])
      graph.edges[range(number_of_edges + 6, number_of_edges + 10)].data['direction'] = th.tensor([[[0], [1]]] * 4, device=self.device)

      graph.edges[range(number_of_edges, number_of_edges + 10)].data['frequency'] = th.tensor(np.array([[1.]] * 10), device=self.device)
    return hidden_states  

def load_from_file(filename, embedding_size, include_frequency=False, graph_embedding_type='candidates'):
  filename = filename if filename[-4:] == '.pth' else f'{filename}.pth'
  # modelname = filename.split('/')[-1].split('.')[0]
  # if 'full' in modelname:
  #   graph_embedding_type = 'full'
  # elif 'candidates' in modelname:
  #   graph_embedding_type = 'candidates'
  # elif 'chosen' in modelname:
  #   graph_embedding_type = 'chosen'
  # else:
  #   graph_embedding_type = 'candidates'

  model = GenerativeModel(embedding_size, include_frequency=include_frequency, graph_embedding_type=graph_embedding_type)
  device = th.device('cuda' if th.cuda.is_available() else 'cpu')
  model.load_state_dict(th.load(filename, map_location=device)['state_dict'])
  if th.cuda.is_available():
    model.cuda()
  model.training = False
  model.eval()
  return model
