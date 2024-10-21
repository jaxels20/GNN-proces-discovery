import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import numpy as np
import matplotlib.pyplot as plt
from project.ml_models import utils


def bernoulli_action_log_prob(logit, action):
  """Calculate the log p of an action with respect to a Bernoulli
  distribution. Use logit rather than prob for numerical stability."""
  if action == 0:
    return F.logsigmoid(-logit)
  else:
    return F.logsigmoid(logit)


class GraphEmbedding(nn.Module):
  def __init__(self, node_hidden_size):
    super().__init__()

    self.graph_hidden_size = 2 * node_hidden_size

    self.node_gating = nn.Sequential(
      nn.Linear(node_hidden_size, 1),
      nn.Sigmoid()
    )
    self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)

  def forward(self, hidden_states):
    return (self.node_gating(hidden_states) * self.node_to_graph(hidden_states)).sum(0, keepdim=True)


class AddPlaceAgent(nn.Module):
  def __init__(self, node_hidden_size, graph_embedding_type):
    super().__init__()
    self.graph_embedding_type = graph_embedding_type
    self.graph_embedding = GraphEmbedding(node_hidden_size)
    self.add_place = nn.Linear(self.graph_embedding.graph_hidden_size, 1)

  def prepare_training(self):
    self.log_probabilities = []

  def prepare_inference(self):
    self.probabilities = []
    self.add_probabilities = []

  def forward(self, graph, node_embeddings, action):
    '''
    :param graph:
    :param node_embeddings:
    :param action: 0 for stopping, 1 for adding a place.
    :return: whether to add a new place or not.
    '''
    first_transition_index = max(np.nonzero(graph.ndata['label'].numpy().flatten() == 1)[0])
    first_place_index      = min(np.nonzero(graph.ndata['label'].numpy().flatten() == 0)[0])
    last_place_index       = max(np.nonzero(graph.ndata['silent_np'].numpy().flatten() == 0)[0])

    if self.graph_embedding_type not in ['full', 'chosen', 'candidates']:
      raise ValueError(f'Graph embedding {self.graph_embedding_type} not known, choose one of [full (mws6c), chosen (mws7c), candidates (mws8c)].')

    if self.graph_embedding_type == 'chosen':
      # Only using the chosen nodes (transitions and chosen decisions):
      node_embeddings = th.cat((node_embeddings[range(first_transition_index, first_place_index)], node_embeddings[graph.ndata['decision'].numpy().flatten() == 1]))
    elif self.graph_embedding_type == 'candidates':
      # Only excluding trace nodes and invalid silent transitions:
      node_embeddings = th.cat((node_embeddings[range(first_transition_index, last_place_index + 1)], node_embeddings[graph.ndata['silent_np'].numpy().flatten() == -1]))
    elif self.graph_embedding_type == 'full':
      pass

    graph_embedding = self.graph_embedding(node_embeddings)
    logit = self.add_place(graph_embedding)
    probability = th.sigmoid(logit)

    if not self.training:
      p = max(0., min(1., probability.item()))
      action = int(p > 0.5) #Bernoulli(p).sample().item()
      self.add_probabilities.append(probability.data)
      if action == 0:
        self.probabilities.append(1 - probability.data)
      else:
        self.probabilities.append(probability.data)

    add_place = bool(action == 1)

    if self.training:
      self.log_probabilities.append(bernoulli_action_log_prob(logit, action))

    return add_place, probability


class ChoosePlaceAgent(nn.Module):
  def __init__(self, node_hidden_size):
    super().__init__()
    self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    self.choose_place = nn.Linear(node_hidden_size, 1)

  def prepare_training(self):
    self.log_probabilities = []

  def prepare_inference(self):
    self.probabilities = []

  def forward(self, graph, embeddings, place, number_of_candidates, top=None, return_probs=False):
    n = graph.number_of_nodes()
    place_scores = self.choose_place(embeddings[-number_of_candidates:]).view(1, -1)

    # Set score to very low for already chosen places and not available silent transitions
    place_scores += (graph.nodes[range(n - number_of_candidates, n)].data['silent_np'] == 1) * -1e30
    place_scores += (graph.nodes[range(n - number_of_candidates, n)].data['decision'] == 1) * -1e30
    if not self.training:
      place_scores += (graph.nodes[range(n - number_of_candidates, n)].data['unsound'] == 1) * -1e30

    place_probs = F.softmax(place_scores, dim=1)
    if return_probs:
      return place_probs[0]

    if not self.training:
      if top is not None:
        top_places = th.nonzero((place_probs[0] == sorted(place_probs[0], reverse=True)[top]))
        place = top_places[np.random.choice(len(top_places))].data.item()
      else:
        place = place_probs.shape[1]
        while abs(place) >= place_probs.shape[1]:
          sample = Categorical(place_probs).sample()
          place = sample.item()

      self.probabilities.append(place_probs[0][place].data)

    graph.nodes[n - number_of_candidates + place].data['decision'] = th.tensor([1.], device=self.device)

    if self.training:
      if place_probs.nelement() > 1:
        self.log_probabilities.append(F.log_softmax(place_scores, dim=1)[:, place: place + 1])
        # TODO Keep track of (negative) probabilities on places which cause the model to be unsound (these should be minimized)
        # TODO This could help the model to not choose unsound places.

    return place


class GATLayer(nn.Module):
  def __init__(self, in_dim, out_dim, name=''):
    super().__init__()
    self.name = name
    self.fc  = nn.Linear(in_dim, out_dim, bias=False)
    self.fcr = nn.Linear(in_dim, out_dim, bias=False)
    self.attn_fc  = nn.Linear( 2 * out_dim, 1, bias=False)
    self.attn_fcr = nn.Linear(2 * out_dim, 1, bias=False)
    self.reset_parameters()

  def reset_parameters(self):
    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_normal_(self.fc.weight, gain=gain)
    nn.init.xavier_normal_(self.fcr.weight, gain=gain)
    nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
    nn.init.xavier_normal_(self.attn_fcr.weight, gain=gain)

  def edge_attention(self, edges):
    z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
    zr2 = th.cat([edges.src['zr'], edges.dst['zr']], dim=1)
    a  = self.attn_fc(z2)
    ar = self.attn_fc(zr2)
    return {'e': F.leaky_relu(a), 'er': F.leaky_relu(ar)}

  def message_func(self, edges):
    nrNodes, nrFeatures = edges.src['z'].shape
    q = th.sum(edges.data['direction'] * th.cat((edges.src['z'],  edges.src['zr']),  dim=1).reshape(nrNodes, 2, nrFeatures), dim=1)

    return {
      'q': q,
      'e': th.sum(edges.data['direction'] * th.cat((edges.data['e'], edges.data['er']), dim=1).reshape(nrNodes, 2, 1), dim=1)
    }

  def reduce_func(self, nodes):
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    h = th.sum(alpha * nodes.mailbox['q'], dim=1)
    return {'h': h}

  def forward(self, g, h):
    with g.local_scope():
      z  = self.fc(h)
      zr = self.fcr(h)
      g.nodes[:].data['z']  = z
      g.nodes[:].data['zr'] = zr
      g.apply_edges(self.edge_attention)
      g.update_all(self.message_func, self.reduce_func)
      return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
  def __init__(self, in_dim, out_dim, num_heads, name=''):
    super().__init__()
    self.heads = nn.ModuleList()
    for i in range(num_heads):
      self.heads.append(GATLayer(in_dim, out_dim, name=f'{name}_head_{i}'))

  def forward(self, g,  h):
    head_outs = [attn_head(g, h) for attn_head in self.heads]
    return th.cat(head_outs, dim=1)
