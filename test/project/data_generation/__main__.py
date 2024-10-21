from project.data_generation.generator import Generator
from project.data_handling.petrinet import getPetrinetFromFile

import argparse
import json
import os
import time
import numpy as np

np.set_printoptions(linewidth=400)

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='Verbose output', action="store_true")
parser.add_argument('-n', '--numberOfTraces', help='Number of traces', type=int)
parser.add_argument('-c', '--config', help='Config filename in json format', type=str)
parser.add_argument('-d', '--dataDirectory', help='dataDirectory', type=str)
parser.add_argument('-pf', '--petrinetFilename', help='petrinetFilename', type=str)

args = parser.parse_args()

datasetName = args.petrinetFilename.split('/')[-1].split('.')[0]

print(datasetName)
print(args.petrinetFilename)
petrinet, initialMarking, finalMarking, stochasticInformation = getPetrinetFromFile(fPetrinetName=args.petrinetFilename, fVisualize=True)
print(stochasticInformation)

generator = Generator(datasetName, petrinet, initialMarking, finalMarking, stochasticInformation)
generator.generateData(args.numberOfTraces, fDirectory=args.dataDirectory, fVerbose=args.verbose)
