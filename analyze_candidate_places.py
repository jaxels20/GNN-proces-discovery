from src.CandidateAnalyzer import CandidateAnalyzer
SCENARIO = "controlled"

if __name__ == "__main__":
    if SCENARIO == "controlled":
        input_dir = "./controlled_scenarios/"
        output_dir = "./candidate_analysis_results/controlled_scenarios/"
        analyzer = CandidateAnalyzer(input_dir, output_dir)
        analyzer.evaluate_on_controlled_scenarios()
        print("Candidate analysis complete.")     
    elif SCENARIO == "synthetic":
        input_dir = "./data_generation/synthetic_data/train/"
        output_dir = "./candidate_analysis_results/synthetic_data/"
        analyzer = CandidateAnalyzer(input_dir, output_dir)
        analyzer.evaluate_on_synthetic_data()
        print("Candidate analysis complete.")