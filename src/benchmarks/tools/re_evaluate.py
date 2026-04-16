import pandas as pd
from src.core.judge import evaluate_response

INPUT_CSV = "kaggle_results.csv"
OUTPUT_CSV = "kaggle_results_corrected.csv"

def re_evaluate():
    print(f"Loading results from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Re-evaluate both columns
    print("Re-evaluating responses with fixed Judge logic...")
    
    # Apply judge logic to 'response_no_defense'
    # Handle NaN values if any (though send_prompt returns "ERROR" strings usually)
    df['blocked_no_defense'] = df['response_no_defense'].apply(lambda x: evaluate_response(str(x)))
    
    # Apply judge logic to 'response_defense'
    df['blocked_defense'] = df['response_defense'].apply(lambda x: evaluate_response(str(x)))

    # Save corrected results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Corrected results saved to {OUTPUT_CSV}")

    # Calculate new stats
    success_rate_no_defense = (1 - df['blocked_no_defense'].mean()) * 100
    success_rate_defense = (1 - df['blocked_defense'].mean()) * 100

    print(f"\n--- Corrected Summary ({len(df)} samples) ---")
    print(f"Attack Success Rate (No Defense): {success_rate_no_defense:.2f}%")
    print(f"Attack Success Rate (With Defense): {success_rate_defense:.2f}%")

if __name__ == "__main__":
    re_evaluate()
