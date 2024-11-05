from PetriNet import PetriNet
from EventLog import EventLog
from pm4py.algo.evaluation.replay_fitness.algorithm import apply as replay_fitness
from pm4py.algo.evaluation.precision.algorithm import apply as precision
from pm4py.algo.evaluation.generalization.algorithm import apply as generalization
from pm4py.algo.evaluation.simplicity.algorithm import apply as simplicity


class ModelLogEvaluator:
    def __init__(self, proces_model: PetriNet, event_log: EventLog):
        self.process_model = proces_model
        self.event_log = event_log
        
        # convert the process model to pm4py format
        self.process_model_pm4py, self.init_marking, self. final_marking = self.process_model.to_pm4py()

        # convert the eventlog to pm4py format
        self.event_log_pm4py = self.event_log.to_pm4py()
    
    def get_evaluation_metrics(self):
        data = {
            "simplicity": self.get_simplicity(),
            "generalization": self.get_generalization(),
            "replay_fitness": self.get_replay_fitness(),
            "precision": self.get_precision(),
        }
        data["f1_score"] = self.get_f1_score(data["precision"], data["replay_fitness"]['averageFitness'])
        
        return data    
    
    def get_simplicity(self):
        simplicity_value = simplicity(self.process_model_pm4py)
        return simplicity_value
    
    def get_generalization(self):
        generalization_value = generalization(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return generalization_value
    
    def get_replay_fitness(self):
        fitness = replay_fitness(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return fitness
    
    def get_precision(self):
        precision_value = precision(self.event_log_pm4py, self.process_model_pm4py, self.init_marking, self.final_marking)
        return precision_value
    
    def get_f1_score(self, precision=None, fitness=None):
        if precision is None:
            precision = self.get_precision()
        if fitness is None:
            fitness = self.get_replay_fitness()
        
        f1_score = 2 * (precision * fitness) / (precision + fitness)
        return f1_score
        
    
        
        
    
    


    