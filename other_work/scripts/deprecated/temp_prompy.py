import prompy
from pathlib import Path
from prompy import DocumentationGenerator
from prompy import ProMExecutor, ScriptBuilder

import os
import shlex
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from threading import Timer
from pathlib import Path


class ProMExecutor2:
  def __init__(self, parameters):
    self.prom_directory = parameters['prom_directory']
    self.lib_directory = parameters['lib_directory']
    self.dist_directory = parameters['dist_directory']
    self.memory = parameters['memory'] if 'memory' in parameters else '8G'
    self.java = parameters['java'] if 'java' in parameters else 'java'
    self.command = ''

    self.script_output_prefix = '[SCRIPT_OUTPUT] '

  def run_script(self, script, timeout=-1, verbosity=0):
    self.__initialize_command_with_arguments()

    script = ';\n'.join(script.split(';'))

    temp_filename = self.__prepare_script_for_output_parsing(script)

    command = f'{self.command} -f {temp_filename}'
    stdout, stderr = self.__run_command(command, timeout)
    print(stdout)
    # print(stderr)

    output = ''
    if verbosity >= 1:
      output = stderr + '\n'

    output += self.__process_output(stdout, verbosity)

    self.__clean(temp_filename)
    return output

  def __execute_command(self, command, timeout):
    if os.name == 'posix':
      command = shlex.split(command)

    process = Popen(command, shell=False, stdout=PIPE, stderr=STDOUT, universal_newlines=True)
    if timeout <= 0:
      for stdout_line in iter(process.stdout.readline, ""):
        print(stdout_line)
        yield stdout_line
      process.stdout.close()
      process.wait()
    else:
      timer = Timer(timeout, process.kill)
      try:
        timer.start()
        for stdout_line in iter(process.stdout.readline, ""):
          yield stdout_line
        process.stdout.close()
        process.wait()
      finally:
        if not timer.is_alive():
          print('Timeout reached.')

  def __run_command(self, command, timeout=-1):
    current_directory = os.getcwd()
    os.chdir(self.prom_directory)

    lines = []
    for line in self.__execute_command(command, timeout):
      lines.append(line)

    # Reduce initialization lines.
    try:
      index = next(i for i, line in enumerate(lines) if line.startswith('Start plug-in'))
    except StopIteration:
      print('No plug-in has started yet. Increase the timeout if a plug-in is used in the script.')
      index = 0
    stdout = ''.join(lines[index:])
    stderr = ''
    os.chdir(current_directory)
    return stdout, stderr

  def get_available_functions(self):
    # Run the ProM CLI with -l argument returning all the available functions from the loaded plugins.
    self.__initialize_command_with_arguments()
    command = f'{self.command} -l'
    print(command)
    stdout, stderr = self.__run_command(command, 60)
    print(stderr)
    print(stdout)
    lines = stdout.split('\n')
    return [line for line in lines if len(line) > 0 and line[0] != '[' and '(' in line]

  def __prepare_script_for_output_parsing(self, script):
    # Create a temporary file with the final script. Add a prefix to each print command, so it can be parsed accordingly.
    temp_filename = 'TEMP_SCRIPT_FILE.txt'
    with open(temp_filename, 'w') as outFile:
      for line in script.split('\n'):
        outFile.write(line.replace('print(', f'print("{self.script_output_prefix}" + ').replace('System.out.println(',
                                                                                                f'System.out.println("{self.script_output_prefix}" + ') + "\n")
    return Path(os.getcwd()) / temp_filename

  def __process_output(self, output, verbosity=0):
    # Print the output from the script. If verbosity equals 0, only the script output is printed, otherwise all output
    #  from the java command (i.e. ProM CLI output).
    lines = []
    for line in output.split('\n'):
      if line[:len(self.script_output_prefix)] == self.script_output_prefix or verbosity >= 1:
        lines.append(line)
    return '\n'.join(lines)

  def __get_plugins(self):
    plugins = []
    # Load .jar files from dist directory
    dist_directory = self.prom_directory / self.dist_directory
    for entry in os.listdir(dist_directory):
      if os.path.isfile(os.path.join(dist_directory, entry)):
        # Check whether it's .jar file
        plugins.append(self.dist_directory / entry)

    # Load .jar files from lib directory
    lib_directory = self.prom_directory / self.lib_directory
    for entry in os.listdir(lib_directory):
      if os.path.isfile(os.path.join(lib_directory, entry)):
        # Check whether it's .jar file
        plugins.append(self.lib_directory / entry)

    # Join the file strings for the 'classPath' argument in the java command. (On Linux, the paths should be joined by
    #  colons, on Windows by semicolons.
    glue = ':' if os.name == 'posix' else ';'
    slash_type = '/' if os.name == 'posix' else '\\'
    plugins_string = glue.join([f'.{slash_type}{os.fspath(path)}' for path in plugins])
    return plugins_string

  def __initialize_command_with_arguments(self):
    plugins = self.__get_plugins()
    arguments = [
      self.java,
      '-da',
      f'-classpath "{plugins}" '
      f'-Djava.library.path=./{self.lib_directory}',
      f'-Djava.util.Arrays.useLegacyMergeSort=true '
      f'-Xmx{self.memory}',
      'org.processmining.contexts.cli.CLI'
    ]
    self.command = ' '.join(arguments)

  def __clean(self, fTempFilename):
    # Remove the temporary file and reset the commandString.
    os.remove(fTempFilename)
    self.command = ''



if __name__ == '__main__':
  dir = '/mnt/c/Users/s140511/tue/thesis/thesis_data/process_trees_simple_ws2/'

  parameters = {'prom_directory': Path('/mnt/c/Users/s140511/tue/Courses/CapitaSelecta/prom'),
                'lib_directory': Path('lib'), 'dist_directory': Path('dist'), 'memory': '4G', 'java': 'java'}
  prom_executor = ProMExecutor2(parameters)

  prom_executor.get_available_functions()
  print(fdsa)

  script = ScriptBuilder.get_log(f'{dir}/logs/0000.xes')
  script += ScriptBuilder.get_petrinet(f'{dir}/petrinets/0000.pnml')
  script += ScriptBuilder.get_mapping('petrinet', 'log')
  script += '\nresult = replay_a_log_on_petri_net_for_conformance_analysis(petrinet, log, mapping);'
  script += '\nprint("result: " + result);'

  # script += ScriptBuilder.get_fitness('petrinet', 'log', 'mapping', variable_prefix='fitness_')
  # script += 'print(fitness_result.getInfo());\n'
  # script += ScriptBuilder.get_precision('petrinet', 'marking', 'mapping', alignment_result_variable_name='fitness_result', variable_prefix='precision_')
  # script += '\nprint("Precision: " + precision_result.getPrecision());'
  # script += '\nprint("Generalization: " + precision_result.getGeneralization());'
  script += ScriptBuilder.end()



  # script = ScriptBuilder.get_petrinet(dir)
  # script += ScriptBuilder.get_soundness('petrinet')
  # script += '\nprint("Soundness: " + soundness);'
  # script += ScriptBuilder.end()
  print(ScriptBuilder.show_script(script))

  output = prom_executor.run_script(script, timeout=-1, verbosity=2)
  print(output)



