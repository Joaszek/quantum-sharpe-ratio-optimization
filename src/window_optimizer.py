import pandas as pd
from data_loader import load_data
from return_calculator import calculate_logs
from sharpe_optimizer import optimize_sharpe
from sharpe_ratio_utils import calculate_profit

def run_sliding_window_optimization(window_size=20, step_size=5, k=2, lambda_risk=0.4, output_path="results/sliding_window_sharpe_results.csv"):
    df = load_data()
    log_returns = calculate_logs()
    results = []

    max_start = len(log_returns) - window_size
    for start in range(0, max_start + 1, step_size):
        end = start + window_size
        window_returns = log_returns.iloc[start:end]

        selected_assets, objective_value = optimize_sharpe(window_returns, k=k, lambda_risk=lambda_risk)
        realized_profit = calculate_profit(log_returns.iloc[end:end+window_size], selected_assets)

        results.append({
            "window_start": start,
            "window_end": end,
            "selected_assets": selected_assets.tolist(),
            "objective_value": objective_value,
            "realized_profit": realized_profit
        })

        print(f"Window {start}-{end}")
        print("Selected assets:", selected_assets)
        print("Objective value:", objective_value)
        print("Realized profit after window:", realized_profit)
        print("---")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    run_sliding_window_optimization()
