from src.CandidateAnalyzer import CandidateAnalyzer


INPUT_DIR = "./controlled_scenarios/"
OUTPUT_DIR = "./candidate_analysis_results/"


if __name__ == "__main__":
    analyzer = CandidateAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.evaluate_candidate_places_on_all_pairs()
    print("Candidate analysis complete.")