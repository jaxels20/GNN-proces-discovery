from project.process_mining.process_discovery import AlphaMiner, HeuristicsMiner, InductiveMiner, GnnMiner, SplitMiner, GroundTruth, ImportMiner

import argparse
import json
import os
import tqdm

miners = {
  'alpha':      AlphaMiner,
  'heuristics': HeuristicsMiner,
  'inductive':  InductiveMiner,
  'split':      SplitMiner,
  'gnn':        GnnMiner,
  'gt':         GroundTruth,
  'import':     ImportMiner
}

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Config filename in json format', type=str)
parser.add_argument('-d', '--dataDirectory', help='dataDirectory', type=str)
parser.add_argument('-l', '--logFilename', help='logFilename', type=str)
parser.add_argument('-m', '--miner', choices=['alpha', 'heuristics', 'inductive', 'gnn', 'split', 'gt', 'import'], help='Miner to use for discovery')
parser.add_argument('-mf', '--model_filename', help='Filename of the statedict of the AI model', type=str)
parser.add_argument('-v', '--visualize', help='Visualize the petrinet', action="store_true")
parser.add_argument('-e', '--export', help='Export the petrinet', type=str)
parser.add_argument('-cc', '--conformanceChecking', help='Perform conformance checking on discovered model(s)', action="store_true")

parser.add_argument('-esize', '--embedding_size', help='(int, default=21) Size of the initial embeddings.', type=int)
parser.add_argument('-estrat', '--embedding_strategy', help='(str, default=onehot) Embedding strategy, supported strategies are: [onehot, random].', type=int)
parser.add_argument('-exfreq', '--exclude_frequency', help='(bool, default=false) Embed trace frequency in initial embedding.', action="store_true")

parser.add_argument('-bw', '--beam_width', help='Beam width', type=int)
parser.add_argument('-bl', '--beam_length', help='Beam length', type=int)
parser.add_argument('-np', '--number_of_petrinets', help='Number of petrinets to find', type=int)
parser.add_argument('-tx', '--top_x_traces', help='Number of trace variants to take into account', type=int)
parser.add_argument('-ln', '--length_normalization', help='Perform length normalization', type=int)

parser.add_argument('-to', '--timeout', help='Time out', type=int)
parser.add_argument('-prom', '--prom', help='Use prom for discovery of inductive/heuristics miner.', action="store_true")
parser.add_argument('-ipnet', '--import_petrinet', help='Use existing Petri net instead of discovering a new one', type=str)

args = parser.parse_args()


if args.config is None:
  args.config = 'config.json'

with open(f'{os.path.dirname(os.path.realpath(__file__))}/{args.config}') as jsonFile:
  config = json.load(jsonFile)

if args.dataDirectory is None:
  args.dataDirectory = config['dataDirectory']

if args.logFilename is None:
  args.logFilename = config['logFilename']

if args.miner is None:
  args.miner = config['miner']

###################
# Start of script #
###################

if ',' in args.logFilename:
  log_filenames = range(int(args.logFilename.split(',')[0]), int(args.logFilename.split(',')[1]))
else:
  log_filenames = [args.logFilename]

for log_filename in tqdm.tqdm(log_filenames):
  log_filename = str(log_filename)
  print(log_filename)
  if args.miner == 'gnn':
    embedding_size = 21 if args.embedding_size is None else args.embedding_size
    embedding_strategy = 'onehot' if args.embedding_strategy is None else args.embedding_strategy

    discoverer = miners[args.miner](f'{args.dataDirectory}{log_filename}', args.model_filename, embedding_size=embedding_size,
                                    embedding_strategy=embedding_strategy, include_frequency=not args.exclude_frequency)
    options = {
      'beam_width': args.beam_width,
      'beam_length': args.beam_length,
      'number_of_petrinets': args.number_of_petrinets if args.number_of_petrinets is not None else 1,
      'export': '' if args.export is None else args.export,
      'length_normalization': True if args.length_normalization is None else bool(args.length_normalization),
      'timeout': args.timeout
    }
    print(options)
    if args.top_x_traces is not None:
      options['topXTraces'] = args.top_x_traces
  else:
    options = {'export': '' if args.export is None else args.export}
    print(options)
    if log_filename[-4:] == '.npz':
      log_filename = log_filename[:-4]
    discoverer = miners[args.miner](f'{args.dataDirectory}{log_filename}')

  discoverer.discover(conformance_check=args.conformanceChecking, **options, args=vars(args))

  if args.visualize:
    discoverer.visualize()
