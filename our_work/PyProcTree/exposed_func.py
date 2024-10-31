import json
from pathlib import Path
import time
import multiprocessing
from generate_newick_trees import abortable_generate_tree, generate_tree
from generate_logs import write_as_csv, write_as_xes, select_child
import os
from tree import TreeNode
from simulateLog import LogSimulator
from simulateTrace import TraceSimulator
from add_noise import NoiseGenerator



def generate_process_trees(config: json, output_dir: str, timeout: int = 1000, render_image: bool = False):
    start_time = time.time()
    first_line = True
    parameter_lines = open(config)
    population_index = 1

    for line in parameter_lines:
        if first_line:
            first_line = False
            continue
        else:
            line = line.rstrip()
            parameters = line.split(';')

        no_trees = int(parameters[12])

        pool = multiprocessing.Pool(processes=4)
        multiple_results = [pool.apply_async(abortable_generate_tree, args=(line, i,
                                                                            population_index,
                                                                            render_image,
                                                                            output_dir,
                                                                            timeout
                                                                            )) for i in
                            range(1, no_trees + 1)]
        print('generated',sum([res.get() for res in multiple_results]), 'models for population' ,str(population_index))

        population_index += 1

    # print total generation time
    total_time = str(time.time()-start_time)
    print('total tree generation time:', total_time)


def generate_logs(input_dir: str, output_dir: str, size: int, noise: float, timestamps: bool, format: str):
    #specify the folder with the trees
    #tree_files = glob.glob(tree_folder + "*.nw")
    tree_files = os.listdir(input_dir)
    tree_files = [input_dir + "/" + file for file in tree_files if file.endswith('.nw')]
    #for each tree
    for filepath in tree_files:
        #generate traces
        t = TreeNode(filepath,format=1)
        if t.get_tree_root().name == 'choice':
            traces = []
            children = t.get_children()
            for i in range(size):
                child = select_child(children)
                if child.is_leaf():
                    artificial_parent = TreeNode('sequence:1;')
                    artificial_parent.add_child(child=child)
                    simulator = TraceSimulator(artificial_parent.write(format=1,format_root_node=True),
                                            timestamps)
                else:
                    simulator = TraceSimulator(child.write(format=1,format_root_node=True),
                                            timestamps)
                
                traces.append(simulator.case.trace)
        else:
            simulator = LogSimulator(t.write(format=1,format_root_node=True),size, timestamps)
            traces = simulator.returnLog()

        #add noise
        noise_generator = NoiseGenerator(traces, noise)
        traces = noise_generator.resulting_traces

        #write log to csv-file
        tree_index = filepath[filepath.find('_'):filepath.rfind('.nw')]
        if format == 'csv':
            write_as_csv(traces,tree_index,timestamps)
        elif format == 'xes':
            write_as_xes(traces,tree_index,timestamps, output_dir)


if __name__ == '__main__':
    
    #generate_process_trees("./data/parameter_files/example_parameters.csv", "./data/trees/", timeout=10000, render_image=False)
    #generate_logs("./data/trees/", "./data/logs/", 5, 0.0, False, "xes")
    pass
