from src.CandidateAnalyzer import CandidateAnalyzer


INPUT_DIR = "./controlled_scenarios/"
OUTPUT_DIR = "./candidate_analysis_results/controlled_scenarios/"


if __name__ == "__main__":
    analyzer = CandidateAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.evaluate_on_controlled_scenarios()
    print("Candidate analysis complete.")
    


# INPUT_DIR = "./data_generation/synthetic_data/train/"
# OUTPUT_DIR = "./candidate_analysis_results/synthetic_data/"

# if __name__ == "__main__":
#     analyzer = CandidateAnalyzer(INPUT_DIR, OUTPUT_DIR)
#     analyzer.evaluate_on_synthetic_data()
#     print("Candidate analysis complete.")