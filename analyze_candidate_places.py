from src.CandidateAnalyzer import CandidateAnalyzer

if __name__ == "__main__":
    input_dir = "./controlled_scenarios/"
    
    output_dir = "./candidate_analysis_results/experiment_1/"
    analyzer = CandidateAnalyzer(input_dir, output_dir)
    analyzer.evaluate_on_controlled_scenarios()
    print("Candidate analysis complete.")    
    
    output_dir = "./candidate_analysis_results/experiment_2/"
    analyzer = CandidateAnalyzer(input_dir, output_dir)
    analyzer.evaluate_on_controlled_scenarios(add_silent_transitions=True)
    print("Candidate analysis complete.")     