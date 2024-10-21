from project.data_handling.petrinet import PetrinetHandler

import tqdm

dataset = 'simple'
dir = {'simple': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/petrinets',
       'complex': '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_complex_ws/petrinets'}[dataset]

test_samples = {'simple': range(1970, 2627), 'complex': range(1488, 1985)}[dataset]
train_samples = range(test_samples.start)

print('Loading train samples.')
train_pnet_signatures = []
for train_sample in tqdm.tqdm(train_samples):
  pnet = PetrinetHandler()
  pnet.importFromFile(f'{dir}/{train_sample:04d}.pnml')
  train_pnet_signatures.append(pnet.get_signature())

print('Checking test samples')
duplicates = []
for test_sample in tqdm.tqdm(test_samples):
  pnet = PetrinetHandler()
  pnet.importFromFile(f'{dir}/{test_sample:04d}.pnml')
  signature = pnet.get_signature()
  for train_sample, train_pnet_signature in enumerate(train_pnet_signatures):
    if signature == train_pnet_signature:
      print(f'Sample {test_sample} is a duplicate with {train_sample}.')
      print(signature)
      print(train_pnet_signature)
      duplicates.append(test_sample)
      break
print(duplicates)