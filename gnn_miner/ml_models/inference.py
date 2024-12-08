from gnn_miner.data_handling.petrinet import PetrinetHandler, create_petrinet

import torch as th
from colorama import Fore, Style
import time
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import copy
import bisect


class Inference:
  def __init__(self, model):
    self.model = model
    self.model.eval()
    self.model.training = False

  def evaluation_string(self, true_places, predicted_places):
    true_places = sorted(true_places)
    string = ''
    for true_place in true_places:
      if true_place in predicted_places:
        string += Fore.CYAN
      string += f'{true_place}{Style.RESET_ALL} '
    string += Fore.RED
    for predicted_place in predicted_places:
      if predicted_place not in true_places:
        string += f'{predicted_place} '

    return string + Style.RESET_ALL

  def getTree(self, top, depth, routes=None):
    if routes is None or len(routes) == 0:
      routes = [[i] for i in range(top)]
    if len(routes[0]) == depth:
      return routes
    return self.getTree(top, depth, [route + [t] for route in routes for t in range(top)])

  def check_prefixes(self, route, prefixes):
    for prefix in prefixes:
      if route[:len(prefix)] == prefix:
        return True
    return False

  def export_prediction(self, target_places, predicted_places, transition_labels, traces=None, fFilename=''):
    pnetHandler = PetrinetHandler()
    pnetHandler.fromPlaces(predicted_places, transition_labels,
                           fPlaceLabels=['' if p in target_places else '1' for p in predicted_places])
    pnetHandler.fromPlaces(target_places, transition_labels,
                           fPlaceLabels=['' if p in predicted_places else '0' for p in target_places])
    if traces is not None:
      startPlace = pnetHandler.addPlace('>')
      pnetHandler.mInitialMarking[startPlace] += 1
      endPlace = pnetHandler.addPlace('|')
      pnetHandler.mFinalMarking[endPlace] += 1
      maxLength = len(max(traces, key=lambda x: len(x)))
      filler_label = len(transition_labels)
      for trace in traces:
        trace += [filler_label] * (maxLength - len(trace))
        currentNode = startPlace
        for index, transition in enumerate(trace):
          letter = transition_labels[transition + 1] if transition < len(transition_labels) else ''
          nextNode = pnetHandler.addTransition(letter) if index % 2 == 0 else pnetHandler.addPlace(letter)
          pnetHandler.addArc(currentNode, nextNode)
          currentNode = nextNode
        if len(trace) % 2 == 0:
          nextNode = pnetHandler.addTransition('')
          pnetHandler.addArc(currentNode, nextNode)
          currentNode = nextNode
        pnetHandler.addArc(currentNode, endPlace)
    pnetHandler.visualize(fExport=fFilename, fDebug=True)

  def get_top_s_coverable_candidates(self, graph, sorted_candidates, chosen, unsounds, beam_width, number_of_non_candidates):
    top_candidates = []
    for candidate in sorted_candidates:
      candidate = candidate.item()
      if candidate in unsounds:
        continue
      if graph.nodes[number_of_non_candidates + candidate].data['decision'] == 1:
        continue
      if graph.nodes[number_of_non_candidates + candidate].data['silent_np'] == 1:
        continue
      s_coverable, description = self.check_s_coverability(graph, chosen + [candidate])
      if s_coverable:
        top_candidates.append(candidate)
        if len(top_candidates) == beam_width:
          return top_candidates
      elif description == 'ns':
        unsounds.add(candidate)
    return top_candidates

  def predict(self, pb, beam_width=3, beam_length=6, max_number_of_places=20, number_of_petrinets=1, length_normalization=False,
              timeout=None, transitionLabels=None):
    self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    self.transitionLabels = transitionLabels
    graph = pb.mNet
    probs = [{'prob': -np.inf}] * number_of_petrinets
    self.all_places = ['{'.join('}'.join(f'({sorted(input)}, {sorted(output)})'.split(']')).split('[')) for input, output in pb.mPossiblePlaces.keys()]
    number_of_places = len(self.all_places)
    original_graph_size = graph.number_of_nodes()
    end_transition_label = int(max(graph.ndata['label']).numpy())
    number_of_non_candidates = original_graph_size - number_of_places
    Score = namedtuple('Score', ['index', 'top_candidate', 'score', 'h'])

    newwie = 1
    if beam_width is not None and bool(newwie):
      models = [self.model] #[copy.deepcopy(self.model) for _ in range(beam_width)]
      with th.no_grad():
        data = {models[0]: {'unsounds': set(), 'chosen': [], 'score': 0, 'done': False, 'wanttobedone': False, 'graph': graph,
                            'h': models[0].forward_initialize(graph, pb.mFeatures, len(pb.mPossiblePlaces))}}

      # TODO fix output formats.
      timings = []
      all_done = False
      i = 0
      start_time = time.time()
      if beam_width < 0:
        beam_width_ = -beam_width
        min_beamwidth = beam_length
      try:
        while not all_done and i < max_number_of_places:
          timings.append(time.time() - start_time)
          # print(f'TOOK {timings[-1]:.3f} seconds')
          if timeout is not None:
            if sum(timings) > timeout:
              print(f'TIMEOUT {timeout} seconds...')
              break
          start_time = time.time()
          beam_width_ = max(min_beamwidth, beam_width_ - 1) if beam_width < 0 else beam_width
          # print(beam_width_)
          scores = []
          # print(Fore.BLUE, i, Style.RESET_ALL)
          i += 1

          all_done = True
          sorted_scores = []
          for index, model in enumerate(models):
            if not data[model]['done']:
              all_done = False
            else:
              scores.append(Score(index, None, data[model]['score'], data[model]['h']))
              continue

            with th.no_grad():
              add_place, probability = model.add_place_agent(data[model]['graph'], data[model]['h'], None)

              if len(data[model]['chosen']) > 40:
                add_place = False

              connected, in_unconnected_transitions, out_unconnected_transitions = model._check_connectedness(data[model]['graph'], data[model]['chosen'])
              # print(Fore.YELLOW, add_place, probability, connected, Style.RESET_ALL)
              probability = probability.item()
              if not add_place:
                if connected:
                  # print(Fore.YELLOW, 'done', Style.RESET_ALL)
                  data[model]['done'] = True
                  probability = 1 - probability
                elif not data[model]['wanttobedone']:
                  data[model]['wanttobedone'] = True
                  silent_indices = range(number_of_non_candidates + number_of_places, data[model]['graph'].number_of_nodes())
                  data[model]['graph'].nodes[silent_indices].data['silent_np'] = th.tensor([1] * len(silent_indices), device=self.device)
                  # print(Fore.RED, 'want to be done. set silent_np=1 for each and every node', Style.RESET_ALL)

              data[model]['score'] += np.log(probability)
              if data[model]['done']:
                scores.append(Score(index, None, data[model]['score'], data[model]['h']))
                continue

              if len(sorted_scores) >= beam_width_ and sorted_scores[beam_width_ - 1] < -data[model]['score']:
                # print('prune')
                continue

              sorted_candidates, probabilities, h = model.forward_step(data[model]['graph'], data[model]['h'], number_of_non_candidates)
              # Find top <beam_width_> candidates.that result in an s-coverable net.
              number_of_top_candidates = 0
              for candidate in sorted_candidates:
                candidate = candidate.item()
                if candidate in data[model]['unsounds']:
                  continue
                if data[model]['graph'].nodes[number_of_non_candidates + candidate].data['decision'] == 1:
                  continue
                if data[model]['graph'].nodes[number_of_non_candidates + candidate].data['silent_np'] == 1:
                  continue
                if data[model]['wanttobedone'] and len(in_unconnected_transitions) + len(out_unconnected_transitions) > 0:
                  # Only choose places that connect 'unconnected' transitions.
                  in_neighbors, out_neighbors = model._get_neighbors(data[model]['graph'], number_of_non_candidates + candidate)
                  if len(in_neighbors & out_unconnected_transitions) == 0 and len(out_neighbors & in_unconnected_transitions) == 0:
                    # print(Fore.RED, 'Cant choose this one since it doesnt connect to an unconnected transition', Style.RESET_ALL)
                    continue
                elif data[model]['wanttobedone'] and not connected:
                  print(Fore.MAGENTA, 'BFS failed', Style.RESET_ALL)

                s_coverable, description = self.check_s_coverability(data[model]['graph'], data[model]['chosen'] + [candidate])
                if s_coverable: # or data[model]['wanttobedone']:
                  score = float('-inf') if probabilities[candidate] == 0 else data[model]['score'] + np.log(probabilities[candidate])
                  if len(sorted_scores) >= beam_width_ and sorted_scores[beam_width_ - 1] < -score:
                    # print('prune 2')
                    break
                  bisect.insort(sorted_scores, -score)
                  scores.append(Score(index, candidate, score, h))
                  number_of_top_candidates += 1

                  if number_of_top_candidates == beam_width_:
                    break
                elif description == 'ns':
                  data[model]['unsounds'].add(candidate)


          # Find the best <beam_width> model+choice pairs
          comparer = lambda x: x.score / len(data[models[x.index]]['chosen'])**0.7 if length_normalization else lambda x: x.score
          if length_normalization:
            best_scores = sorted(scores, key=lambda x: x.score / (2*len(data[models[x.index]]['chosen']))**0.7, reverse=True)[:beam_width_]
          else:
            best_scores = sorted(scores, key=lambda x: x.score, reverse=True)[:beam_width_]
          new_models = []
          new_data = {}
          for best_score in best_scores:
            chosen_candidate = best_score.top_candidate
            chosen = data[models[best_score.index]]['chosen'] + ([chosen_candidate] if chosen_candidate is not None else [])
            # print(Fore.GREEN, chosen, best_score.score, data[models[best_score.index]]['done'], Style.RESET_ALL)
            if best_score.score == float('-inf'):
              continue
            # Create new model copies if necessary
            new_models.append(copy.deepcopy(models[best_score.index]))

            copy_graph = copy.deepcopy(data[models[best_score.index]]['graph'])
            new_data[new_models[-1]] = {
              'unsounds': set(), 'chosen': chosen, 'score': best_score.score, 'done': data[models[best_score.index]]['done'],
              'wanttobedone': data[models[best_score.index]]['wanttobedone'], 'graph': copy_graph,
              'h': th.cat((best_score.h, copy_graph.nodes[:].data['decision'].reshape(copy_graph.number_of_nodes(), 1)), dim=1)
            }

            if chosen_candidate is not None:
              copy_graph.nodes[number_of_non_candidates + chosen_candidate].data['decision'] = th.tensor([1.], device=self.device)
              h = best_score.h
              if chosen_candidate < number_of_places and not new_data[new_models[-1]]['wanttobedone']:
                h = new_models[-1]._create_silent_transitions(copy_graph, best_score.h, chosen, number_of_places, (copy_graph.number_of_nodes() - number_of_non_candidates), pb.mAlphaRelations, end_transition_label)
              new_data[new_models[-1]]['h'] = th.cat((h, copy_graph.nodes[:].data['decision'].reshape(copy_graph.number_of_nodes(), 1)), dim=1)
          models = new_models
          data = new_data
      except KeyboardInterrupt:
        pass

      timings = timings[1:]
      # print(f'Timings per beam search depth: mean {np.mean(timings):.3f}s, median {np.median(timings):.3f}s, min {np.min(timings):.3f}s, max {np.max(timings):.3f}s')

      results = []
      for result in data.values():
        places = [self.all_places[i] for i in result['chosen'] if i < number_of_places]
        place_indices = [i for i, v in enumerate(result['chosen']) if v < number_of_places]
        silent_transition_indices = [i - number_of_places for i in result['chosen'] if i >= number_of_places]
        silent_transitions = self.find_silent_transitions(places, silent_transition_indices)
        results.append({
          'probability': result['score'],
          'places': places,
          'silent_transitions': silent_transitions,
          'choice_probabilities': [],
          'add_probabilities': [],
          'predicted_place_indices': result['chosen']
        })
      return results

    if beam_width is None:
      routes = [None]
    else:
      routes = sorted(self.getTree(beam_width, beam_length))
      routes = [route + [0] * (max_number_of_places - beam_length) for route in routes]
      print(routes)
    prefixes = []

    for route in (tqdm(routes) if len(routes) > 1 else routes):
      print(route)
      if self.check_prefixes(route, prefixes):
        print('can skip this one')
        continue
      with th.no_grad():
        predicted_place_indices_try, prob_try, pruned_prefix, choice_probs, add_probs = \
          self.model(graph, pb.mFeatures, len(pb.mPossiblePlaces), pb.mAlphaRelations, route=route, prune=probs[-1]['prob'], max_number_of_places=max_number_of_places, length_normalization=length_normalization, check_s_coverability_callback=self.check_s_coverability)
        print(predicted_place_indices_try, prob_try)
      prob_try = prob_try.item()
      self.model.clean_graph(graph, original_graph_size)

      if pruned_prefix is not None:
        prefixes.append(pruned_prefix)
      if prob_try >= probs[-1]['prob']:
        probs[-1] = {
          'prob': prob_try,
          'predicted_place_indices': predicted_place_indices_try,
          'route': ''.join([str(r) for r in route]) if route is not None else None,
          'choice_probs': choice_probs,
          'add_probs': add_probs
        }
        probs = sorted(probs, reverse=True, key=lambda x: x['prob'])

    number_of_places = len(self.all_places)

    results = []
    for result in probs:
      print(result)
      if 'predicted_place_indices' not in result:
        results.append({'probability': result['prob'], 'places': [], 'silent_transitions': [], 'choice_probabilities': [], 'add_probabilities': [], 'predicted_place_indices': [] })
        print('Something went wrong, "predicted_place_indices" not found in result.')
        continue
      places = [self.all_places[i] for i in result['predicted_place_indices'] if i < number_of_places]
      place_indices = [i for i, v in enumerate(result['predicted_place_indices']) if v < number_of_places]
      silent_transition_indices =  [i - number_of_places for i in result['predicted_place_indices'] if i >= number_of_places]
      silent_transitions = self.find_silent_transitions(places, silent_transition_indices)
      result['choice_probs'] = [choice_prob for index, choice_prob in enumerate(result['choice_probs']) if index in place_indices]
      result['add_probs']    = [choice_prob for index, choice_prob in enumerate(result['add_probs']) if index in place_indices] + [result['add_probs'][-1]]
      results.append({
        'probability': result['prob'],
        'places': list(set(places)),
        'silent_transitions': list(set(silent_transitions)),
        'choice_probabilities': result['choice_probs'],
        'add_probabilities': result['add_probs'],
        'predicted_place_indices': result['predicted_place_indices']
      })

    return results

  def check_s_coverability(self, graph, predicted_place_indices):
    number_of_places = len(self.all_places)
    places = [self.all_places[i] for i in predicted_place_indices if i < number_of_places]
    silent_transition_indices = [i - number_of_places for i in predicted_place_indices if i >= number_of_places]
    silent_transitions = self.find_silent_transitions(places, silent_transition_indices)
    petrinet_handler = PetrinetHandler()
    petrinet_handler.fromPlaces(places, self.transitionLabels, None, fSilentTransitions=silent_transitions)
    petrinet_handler.remove_loose_transitions()
    petrinet_handler.removeStartAndEndTransitions()

    if len(petrinet_handler.mPetrinet.places) < 2:
      return True, ''

    p_t_arcs, t_p_arcs, place_order, initial_place_indices = petrinet_handler.get_arcs_in_order(places_sort='bfs', transitions_sort='bfs')
    if p_t_arcs is None:
      return False, 'nc'

    transitions = sorted(list(set([arc[1] for arc in p_t_arcs]).union(set([arc[0] for arc in t_p_arcs]))))
    sound, report = create_petrinet(place_order, transitions, p_t_arcs, t_p_arcs, initial_place_indices, short_circuit=False)
    return sound, 'ns'

  def find_silent_transitions(self, places, silent_transition_indices):
    if len(silent_transition_indices) == 0:
      return []
    silent_transitions = []
    added_per_place = [(i - 1)*2 for i in range(1, len(places) + 1)]
    added_total_per_place = [sum(added_per_place[:i + 1]) for i in range(len(places))]
    connections = sum([list(range(i)) for i in range(1, len(places) + 1)], [])
    for silent_transition_index in silent_transition_indices:
      connection_1, count = next(((i, c) for i, c in enumerate(added_total_per_place) if c > silent_transition_index), (None, None))
      connection_2 = connections[int(silent_transition_index / 2.)]
      s = (places[connection_2], places[connection_1]) if (silent_transition_index % 2 == 0) else (places[connection_1], places[connection_2])
      silent_transitions.append(s)
    return silent_transitions

  def evaluate(self, pb, beam_width=3, beam_length=6, max_number_of_places=20, number_of_petrinets=1):
    start_time = time.time()
    graph = pb.mNet
    places = [i for i, p in enumerate(pb.mTarget) if p == 1]
    probs = [{'prob': 0}] * number_of_petrinets

    all_places = [str((set(sorted(input)), set(sorted(output)))) for input, output in pb.mPossiblePlaces.keys()]

    routes = sorted(self.getTree(beam_width, beam_length))
    routes = [route + [0] * (max_number_of_places - beam_length) for route in routes]
    prefixes = []
    for route in routes:
      if self.check_prefixes(route, prefixes):
        continue
      predicted_place_indices_try, prob_try, pruned_prefix = \
        self.model(graph, pb.mFeatures, len(pb.mTarget), route=route, prune=probs[-1]['prob'], max_number_of_places=max_number_of_places)
      if pruned_prefix is not None:
        prefixes.append(pruned_prefix)
      if prob_try > probs[-1]['prob']:
        probs[-1] = {
          'prob': prob_try.item(),
          'predicted_place_indices': predicted_place_indices_try,
          'route': ''.join([str(r) for r in route])}
        probs = sorted(probs, reverse=True, key=lambda x: x['prob'])

    for i in range(len(probs)):
      correct = set(places) == set(probs[i]['predicted_place_indices'])
      correct_with_fps = len(set(places) - set(probs[i]['predicted_place_indices'])) == 0
      predicted_places = [all_places[i] for i in probs[i]['predicted_place_indices']]
      evaluation_string = self.evaluation_string(pb.mTargetPlaces, predicted_places)
      print(f'{Fore.CYAN if correct else Fore.GREEN if correct_with_fps else ""} ({pb.mName:>7}) ({probs[i]["prob"]:.4f})'
            f'{evaluation_string} {Style.RESET_ALL if correct else ""}({time.time() - start_time:.1f}s) {probs[i]["route"]}')

    return probs
