import argparse
import numpy as np

from project.ml_models.preprocessing import get_data
from project.ml_models.model_generative import GenerativeModel, load_from_file
from project.ml_models.training import Training

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='(bool) Verbose output.', action="store_true")
parser.add_argument('-d', '--data_directory', help='(str) Dir where the data is stored, should contain subdirs \'logs_compressed\' and \'petrintes\'.', type=str)
parser.add_argument('-md', '--model_directory', help='(str) Dir to store the gnn model.', type=str)
parser.add_argument('-cpd', '--checkpoint_directory', help='(str default=model_directory/../checkpoints) Dir to store the gnn model checkpoints.', type=str)
parser.add_argument('-mf', '--model_name', help='(str) Name of the gnn model.', type=str)
parser.add_argument('-fc', '--from_checkpoint', help='(bool) Start from scratch (false) or retrieve last saved model (true).', action="store_true")

parser.add_argument('-esize', '--embedding_size', help='(int, default=21) Size of the initial embeddings.', type=int)
parser.add_argument('-estrat', '--embedding_strategy', help='(str, default=onehot) Embedding strategy, supported strategies are: [onehot, random].', type=int)
parser.add_argument('-exfreq', '--exclude_frequency', help='(bool, default=false) Exlude trace frequency from initial embedding.', action="store_true")

parser.add_argument('-tts', '--train_test_split', help='(float, 0.75=true) Train test split ratio.', type=float)
parser.add_argument('-ne', '--number_of_epochs', help='(int default=250) Number of epochs.', type=int)
parser.add_argument('-evale', '--evaluation_epoch', help='(int default=0) Epoch to start evaluation from.', type=int)
parser.add_argument('-ep', '--export_prediction', help='(bool) Export the predictions.', action="store_true")
parser.add_argument('-st', '--skip_training', help='(bool) Skip training, only evaluate.', action="store_true")


args = parser.parse_args()

embedding_size = 21 if args.embedding_size is None else args.embedding_size
embedding_strategy = 'onehot' if args.embedding_strategy is None else args.embedding_strategy

logPrefix      = f'{args.data_directory}/logs_compressed/'
petrinetPrefix = f'{args.data_directory}/petrinets/'


# TODO get all this information from .pth file if from_checkpoint == True.
data_options = {
  'fDepth': 1,
  'fLogPrefix': logPrefix,
  'fEmbeddingSize': embedding_size,
  'fVisualize': False,
  'prepare_petrinet': True,
  'fPetrinetPrefix': petrinetPrefix,
  'embedding_strategy': embedding_strategy,
  'include_frequency': not args.exclude_frequency
}

train_test_split = 0.75 if args.train_test_split is None else args.train_test_split
if train_test_split < 0 or train_test_split > 1:
  raise ValueError(f'train_test_split should be between 0 and 1, is {train_test_split}.')
number_of_samples = len(os.listdir(petrinetPrefix))
train_samples = int(train_test_split * number_of_samples)

print(f'Train samples range from 0 to {train_samples}.')
print(f'Test samples range from {train_samples} to {number_of_samples}.')

perc_to_use = 0.01 # For testing purposes, you could lower this number.

train_graphs = get_data(list(range(train_samples))[:int(perc_to_use*train_samples)], **data_options)
print(f'Number of train samples after loading: {len(train_graphs.keys())} (samples are removed when candidate places do not contain all true places).')
test_graphs = get_data(list(range(train_samples, number_of_samples))[:int(perc_to_use*(number_of_samples - train_samples))], **data_options)
print(f'Number of test samples after loading: {len(test_graphs.keys())} (samples are removed when candidate places do not contain all true places).')

test_graphs = {}

model_filename = f'{args.model_directory}/{args.model_name}.pth'
checkpoints = f'{args.model_directory}/../checkpoints' if args.checkpoint_directory is None else args.checkpoint_directory
checkpoint_name = f'{checkpoints}/{args.model_name}'

if not args.from_checkpoint:
  model = GenerativeModel(embedding_size, include_frequency=not args.exclude_frequency, graph_embedding_type='candidates')
  best_epoch = 0
  sum_loss = np.inf
  losses = []
  best_loss = np.inf
else:
  print('Loading checkpoint.')
  modelcheckpoints = [cp for cp in os.listdir(checkpoints) if '_'.join(cp.split('_')[:-2]) == args.model_name and cp[-4:] == '.pth']
  if len(modelcheckpoints) == 0:
    raise ValueError(f'No checkpoint for {args.model_name} in dir ')

  model_filename = sorted(modelcheckpoints, key=lambda s: int(s.split('_')[-2]), reverse=True)[0]
  print('model_filename', model_filename)

  best_epoch = int(model_filename.split('_')[-2])
  model = load_from_file(f'{checkpoints}/{model_filename}', embedding_size, include_frequency=not args.exclude_frequency)
  with open(f'{checkpoints}/{args.model_name}_train_stats.json', 'r') as file:
    losses = json.load(file)[:best_epoch + 1]
  best_loss = np.mean([v['loss'] for v in losses[best_epoch]])
  print(f'Best epoch was {best_epoch} with a avg loss of {best_loss}.')


number_of_epochs = 250 if args.number_of_epochs is None else args.number_of_epochs
start_evaluation_epoch = 100 if args.evaluation_epoch is None else args.evaluation_epoch
training = Training(model)
training.train_generative(train_graphs,  number_of_epochs=number_of_epochs, test_pbs=test_graphs, dont_train=args.skip_training,
                          save_statistics=checkpoint_name, best_epoch=best_epoch, best_loss=best_loss, train_statistics=losses,
                          export_prediction=args.export_prediction, start_evaluation_epoch=start_evaluation_epoch, start_save_epoch=1)

