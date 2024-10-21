from project.data_handling.petrinet import PetrinetHandler
from project.ml_models.inference import Inference

from colorama import Fore, Style
import torch as th
import numpy as np
import random
import tqdm
import json


class Training:
  def __init__(self, model):
    self.model = model
    self.cuda = th.cuda.is_available()

  def get_evaluation_string(self, true_places, predicted_places):
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

  def get_statistics(self, true_places, predicted_places):
    tp = len(true_places.intersection(predicted_places)) / len(true_places)
    fp = 1 - tp
    fn = len(predicted_places - true_places) / len(true_places)
    return {'tp': tp, 'fp': fp, 'fn': fn}


  def evaluate(self, pb, export_prediction=False):
    inference = Inference(self.model)

    result = inference.predict(pb, beam_width=1, beam_length=1, max_number_of_places=40, number_of_petrinets=1, length_normalization=False)[0]

    self.model.train()
    self.model.training = True

    statistics = {'id': pb.mName, 'true': str(set(pb.place_indices)), 'prediction': str(set(result['predicted_place_indices'])),
                  **self.get_statistics(set(pb.place_indices), set(result['predicted_place_indices']))}
    evaluation_string = self.get_evaluation_string(pb.place_indices, result['predicted_place_indices'])

    if export_prediction:
      petrinet_handler = PetrinetHandler()
      transition_names = [name for name in pb.mTransitionNames if name is not None]
      petrinet_handler.fromPlaces(result['places'], transition_names, fSilentTransitions=result['silent_transitions'])
      petrinet_handler.merge_initial_final_marking()
      petrinet_handler.export(f'{pb.mLogDirectory}predictions/{int(pb.mName):04d}.pnml')
      try:
        petrinet_handler.visualize(fDebug=False, fExport=f'{pb.mLogDirectory}predictions/pngs/{pb.mName}.png')
      except OSError:
        print('OSError')

    return statistics, evaluation_string


  def train_generative(self, train_pbs, number_of_epochs, test_pbs=dict(), start_evaluation_epoch=0, best_epoch=0, best_loss=np.inf,
                       train_statistics=None, export_prediction=False, start_save_epoch=np.inf, save_statistics='', dont_train=False):
    if self.cuda:
      self.model.cuda()
    optimizer = th.optim.Adam(self.model.parameters(), lr=5e-4)

    train_pbs = list(train_pbs.values())
    test_pbs = list(test_pbs.values())
    numberOfPbs = len(train_pbs)
    numberOfTestPbs = len(test_pbs)

    self.model.training = True
    self.model.train()
    bestLoss  = best_loss
    bestEpoch = best_epoch
    prev_trained_epochs = bestEpoch

    skip_datapoints = []
    skip_test_datapoints = []

    training_statistics = train_statistics if train_statistics is not None else []
    test_statistics = []

    for epoch in tqdm.trange(number_of_epochs):
      trained_epochs = epoch + prev_trained_epochs

      epoch_train_statistics = []
      epoch_test_statistics = []
      sumLoss = 0
      random.shuffle(train_pbs)
      pbCount = 0
      for pb in train_pbs:
        if epoch == 0:
          print(pb.mName)
        if pb.mName in skip_datapoints:
          continue

        original_size = pb.mNet.number_of_nodes()

        if not dont_train:
          loss = self.model(pb.mNet, pb.mFeatures, len(pb.mTarget), pb.mAlphaRelations, places=pb.place_indices)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        else:
          with th.no_grad():
            loss = self.model(pb.mNet, pb.mFeatures, len(pb.mTarget), pb.mAlphaRelations, places=pb.place_indices)

        if loss.item() >= 10000000000:
          skip_datapoints.append(pb.mName)
          print(f'epoch: {epoch:03d} ({pbCount:03d}/{numberOfPbs - len(skip_datapoints)}) {loss:>8.4f} ({len(pb.mPossiblePlaces):>3}p) ({pb.mName:>4}) ({"s" if pb.has_silent else "n"}).'
                f'Removing this one from the training dataset.')
          self.model.clean_graph(pb.mNet, original_size)
          continue

        sumLoss += loss.item()
        self.model.clean_graph(pb.mNet, original_size)

        epoch_train_statistics.append({'loss': loss.item()})
        if epoch >= start_evaluation_epoch:
          statistics, evaluation_string = self.evaluate(pb, export_prediction=export_prediction)
          correct = statistics['tp'] == 1 and statistics['fn'] == 0

          epoch_train_statistics[-1] = ({**epoch_train_statistics[-1], **statistics})

          print(f'{Fore.CYAN if correct else ""}epoch: {epoch:03d} ({pbCount:03d}/{numberOfPbs - len(skip_datapoints)}) {loss:>8.4f} ({len(pb.mPossiblePlaces):>3}p) ({pb.mName:>4}) ({"s" if pb.has_silent else "n"}) '
                f'{evaluation_string} {Style.RESET_ALL if correct else ""}')
        else:
          print(f'epoch: {epoch:03d} ({pbCount:03d}/{numberOfPbs - len(skip_datapoints)}) {loss:>8.4f} ({len(pb.mPossiblePlaces):>3}p) ({pb.mName:>4}) ({"s" if pb.has_silent else "n"}) ')
        pbCount += 1

      # Average loss instead of sum.
      sumLoss = sumLoss / pbCount if pbCount != 0 else np.inf
      print(f'epoch: {epoch:03d} done, average loss: {sumLoss:.3f}.')
      if sumLoss < bestLoss:
        print(f'New best found, previous best was epoch {bestEpoch} with average loss {bestLoss:.3f}.')
        bestLoss = sumLoss
        bestEpoch = trained_epochs
        if trained_epochs >= start_save_epoch:
          print('Saving new best.')
          # TODO: save the other layers' information as well and model parameters, like embedding strategy, graph_embedding_type and include_frequency.
          th.save({'state_dict': self.model.state_dict(), 'embedding_size': self.model.embedding_size},
                  f'{save_statistics}_{trained_epochs:03d}_.pth')
      else:
        print(f'Not a new best, previous best was epoch {bestEpoch} with loss {bestLoss:.3f}.')

      training_statistics.append(epoch_train_statistics)

      if save_statistics != '':
        with open(f'{save_statistics}_train_stats.json', 'w', encoding='utf-8') as jsonFile:
          json.dump(training_statistics, jsonFile, sort_keys=True, indent=2)

      if epoch >= start_evaluation_epoch:
        pbCount = 0
        for pb in test_pbs:
          if pb.mName in skip_test_datapoints:
            continue

          original_size = pb.mNet.number_of_nodes()

          with th.no_grad():
            loss = self.model(pb.mNet, pb.mFeatures, len(pb.mTarget), pb.mAlphaRelations, places=pb.place_indices)
            self.model.clean_graph(pb.mNet, original_size)

            if loss.item() >= 10000000000:
              skip_test_datapoints.append(pb.mName)
              print(f'epoch: {epoch:03d} ({pbCount:03d}/{numberOfTestPbs - len(skip_test_datapoints)}) {loss:>8.4f} ({len(pb.mPossiblePlaces):>3}p) ({pb.mName:>4}) ({"s" if pb.has_silent else "n"}) ')
              continue

          statistics, evaluation_string = self.evaluate(pb, export_prediction=export_prediction)
          correct = statistics['tp'] == 1 and statistics['fn'] == 0

          epoch_test_statistics.append({'loss': loss.item(), **statistics})

          print(f'{Fore.CYAN if correct else ""}epoch: {epoch:03d} ({pbCount:03d}/{numberOfTestPbs - len(skip_test_datapoints)}) ({len(pb.mPossiblePlaces):>3}p) ({pb.mName:>4}) ({"s" if pb.has_silent else "n"}) '
                f'{evaluation_string} {Style.RESET_ALL if correct else ""}')
          pbCount += 1

        test_statistics.append(epoch_test_statistics)

        if save_statistics != '':
          with open(f'{save_statistics}_test_stats.json', 'w', encoding='utf-8') as jsonFile:
            json.dump(test_statistics, jsonFile, sort_keys=True, indent=2)

        print(f'best epoch: {bestEpoch}')
