# Quantum Sharpe Ratio Optimization

This project demonstrates portfolio optimization using the Sharpe ratio, leveraging quantum algorithms (QAOA via Qiskit) to select the optimal subset of assets. The workflow includes data loading, return calculation, quantum optimization, and visualization of results.

## Project Structure

- `data/`: Contains historical price data for Apple, CocaCola, and Google.
- `src/`: Source code for data processing, optimization, and plotting.
- `results/`: Stores results from the sliding window optimization.
- `plots/`: Output plots visualizing optimization results.

## Main Components

- **Data Loading:**  
  [`src/data_loader.py`](src/data_loader.py) loads and preprocesses the CSV data.

- **Return Calculation:**  
  [`src/return_calculator.py`](src/return_calculator.py) computes log returns for the assets.

- **Sharpe Ratio Optimization:**  
  [`src/sharpe_optimizer.py`](src/sharpe_optimizer.py) formulates and solves the asset selection problem using QAOA.

- **Sliding Window Optimization:**  
  [`src/window_optimizer.py`](src/window_optimizer.py) runs the optimization over rolling windows and saves results.

- **Result Visualization:**  
  [`src/plot_results.py`](src/plot_results.py) generates plots for objective value, realized profit, and selected assets over time.
