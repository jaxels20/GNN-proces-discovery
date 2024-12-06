import pm4py.algo.simulation.tree_generator.variants.ptandloggenerator as ptandloggenerator
import pm4py.algo.simulation.playout.process_tree.variants.basic_playout as basic_playout
import pm4py.write as pm4pywrite
from multiprocessing import Pool
import os
import json

CONFIG_FILE = "./data_generation/params.json"
INPUT_DIR = "./data_generation/synthetic_data/"
OUTPUT_DIR = "./data_generation/synthetic_data/"
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1



def generate_single_tree(parameters):
    return ptandloggenerator.GeneratedTree(parameters).generate()


def write_process_tree(pt, output_dir, index, assigned_group):
    # create a directory for the group if it does not exist
    output_dir = os.path.join(output_dir, assigned_group)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"pt_{index}.ptml")
    pm4pywrite.write_ptml(pt, filename)


def generate_single_log(tree, parameters):
    return basic_playout.apply(tree, parameters)


def write_event_log(eventlog, output_dir, index, assigned_group):
    # create a directory for the group if it does not exist
    output_dir = os.path.join(output_dir, assigned_group)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"log_{index}.xes")
    pm4pywrite.write_xes(eventlog, filename)


def pt_apply(parameters, num_cores):
    """
    Multiprocessing version of apply method from pm4py library for process tree generation.

    Parameters
    --------------
    parameters
        Parameters of the algorithm, according to the paper:
        - Parameters.MODE: most frequent number of visible activities
        - Parameters.MIN: minimum number of visible activities
        - Parameters.MAX: maximum number of visible activities
        - Parameters.SEQUENCE: probability to add a sequence operator to tree
        - Parameters.CHOICE: probability to add a choice operator to tree
        - Parameters.PARALLEL: probability to add a parallel operator to tree
        - Parameters.LOOP: probability to add a loop operator to tree
        - Parameters.OR: probability to add an or operator to tree
        - Parameters.SILENT: probability to add silent activity to a choice or loop operator
        - Parameters.DUPLICATE: probability to duplicate an activity label
        - Parameters.NO_MODELS: number of trees to generate from model population
    """
    if parameters is None:
        parameters = {}

    parameters.setdefault("mode", 20)
    parameters.setdefault("min", 10)
    parameters.setdefault("max", 30)
    parameters.setdefault("sequence", 0.25)
    parameters.setdefault("choice", 0.25)
    parameters.setdefault("parallel", 0.25)
    parameters.setdefault("loop", 0.25)
    parameters.setdefault("or", 0.0)
    parameters.setdefault("silent", 0.2)
    parameters.setdefault("duplicate", 0)
    parameters.setdefault("no_models", 1)

    no_models = parameters["no_models"]
    if no_models == 1:
        return [generate_single_tree(parameters)]
    else:
        # multiprocessing of tree generation if no_models > 1
        with Pool(num_cores) as pool:
            trees = pool.map(generate_single_tree, [parameters] * no_models)
        return trees


ptandloggenerator.apply = (
    pt_apply  # Overwrite the apply method in ptandloggenerator module
)


def generate_process_trees(
    output_dir: str, parameters: dict, num_cores: int = os.cpu_count() - 2
):
    """Generate process trees and write them to files.

    Args:
        output_dir (str): directory to save the process trees
        parameters (dict): parameters for process tree generation
        num_cores (int, optional): number of cores to utilize in multiprocessing. Defaults to os.cpu_count()-2.

    Returns:
        list: generated process trees
    """
    pts = ptandloggenerator.apply(parameters, num_cores)

    # Assign each process tree to a group ( can be training/validation/test) 
    num_trees = len(pts)
    num_train = int(num_trees * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    num_val = int(num_trees * VALIDATION_SPLIT)
    num_test = num_trees - num_train - num_val
    assigned_groups = ["train"] * num_train + ["validation"] * num_val + ["test"] * num_test

    with Pool(num_cores) as pool:
        pool.starmap(
            write_process_tree, [(pt, output_dir, i, assigned_group) for i, (pt, assigned_group) in enumerate(zip(pts, assigned_groups))]
        )

    return pts


def generate_logs(
    trees: list, output_dir: str, parameters: dict, num_cores: int = os.cpu_count() - 2
):
    """Generate event logs from process trees and write them to files.

    Args:
        trees (list): list of process trees
        output_dir (str): directory to save the event logs
        parameters (dict): parameters for event log generation: {num_traces: int}
        num_cores (int, optional): _description_. Defaults to os.cpu_count()-2.
    """
    with Pool(num_cores) as pool:
        event_logs = pool.starmap(
            generate_single_log, [(tree, parameters) for tree in trees]
        )
        # split the data into training, validation and test sets
        num_event_logs = len(event_logs)
        num_train = int(num_event_logs * (1 - VALIDATION_SPLIT - TEST_SPLIT))
        num_val = int(num_event_logs * VALIDATION_SPLIT)
        num_test = num_event_logs - num_train - num_val
        assigned_groups = ["train"] * num_train + ["validation"] * num_val + ["test"] * num_test
                
        pool.starmap(
            write_event_log,
            [(eventlog, output_dir, i, assigned_group) for i, (eventlog, assigned_group) in enumerate(zip(event_logs, assigned_groups))],
        )


def load_parameters(file_path):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config["tree_generation"], config["log_generation"]


if __name__ == "__main__":
    # check if the input directory exists
    os.makedirs(INPUT_DIR, exist_ok=True)
    tree_gen_config, log_gen_config = load_parameters(CONFIG_FILE)
    pts = generate_process_trees(OUTPUT_DIR, tree_gen_config)
    generate_logs(pts, OUTPUT_DIR, log_gen_config)

    print("Data generation completed.")
