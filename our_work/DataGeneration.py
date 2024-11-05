import pm4py.algo.simulation.tree_generator.variants.ptandloggenerator as ptandloggenerator
import pm4py.algo.simulation.playout.process_tree.algorithm as pt_playout
import pm4py.algo.simulation.playout.process_tree.variants.basic_playout as basic_playout 
import pm4py.write as pm4pywrite

def generate_process_trees(output_dir: str, parameters: dict):
    pts = ptandloggenerator.apply(parameters)
    #export process trees
    for i, pt in enumerate(pts):
        pm4pywrite.write_ptml(pt, output_dir + "pt_" + str(i) + ".ptml")

    return pts
    
def generate_logs(trees: list, output_dir: str, parameters: dict):
    for (i, tree) in enumerate(trees):
        eventlog = basic_playout.apply(tree, parameters)
        # export event logs
        pm4pywrite.write_xes(eventlog, output_dir + "log_" + str(i) + ".xes")

if __name__ == "__main__":
    tree_gen_config = {
        "sequence" : 0.2,
        "choice" : 0.2,
        "parallel" : 0.2,
        "loop" : 0.2,
        "or" : 0.2,
        "mode" : 20,
        "min" : 10,
        "max" : 30,
        "silent" : 0.2,
        "duplicate" : 0,
        "no_models": 100,
    }
    log_gen_config = {
        "num_traces" : 100,
    }
    
    pts = generate_process_trees("./synthetic_data/", tree_gen_config)
    
    generate_logs(pts, "./synthetic_data/", log_gen_config)
    
    

    
    
    
    



