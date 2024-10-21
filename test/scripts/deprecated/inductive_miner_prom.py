from pathlib import Path

parameters = {
  'prom_directory': Path('C:/Users/domin/Documents/prom'), # Directory to the ProM installation.
  'lib_directory':  Path('lib'),   # Lib directory, to be contained in the promDirectory, specified above.
  'dist_directory': Path('dist'), # Dist directory, to be contained in the promDirectory, specified above.
  'memory': '4G', # Memory for Java to use.
  'java': 'jre8\\bin\\java'  # Java command: 'java' for Linux, 'jre8\\bin\\java' for Windows.
}

from prompy import DocumentationGenerator
DocumentationGenerator.generate_documentation('Documentation.py', parameters)
from prompy import ProMExecutor, ScriptBuilder

prom_executor = ProMExecutor.ProMExecutor(parameters)

import os

script = ScriptBuilder.mine('C:/Users/domin/Documents/prompy/examples/log_1.xes',
                            'inductive', {'parameters': 'imf'})
script += ScriptBuilder.get_soundness('petrinet')
script += 'print("Soundness: " + soundness);'
script += ScriptBuilder.get_conformance('petrinet', 'marking', 'log')
script += 'print(_fitness_result.getInfo());'
script += 'print("Precision: " + _precision_result.getPrecision());'
script += 'print("Generalization: " + _precision_result.getGeneralization());'
script += ScriptBuilder.end()

output = prom_executor.run_script(script, timeout=25, verbosity=0)
print(output)