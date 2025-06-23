import numpy as np

def compute_sharpe_ratio(mu, sigma, selection):
    selected_indices = [i for i, bit in enumerate(selection) if bit == 1]
    if not selected_indices:
        return -np.inf

    portfolio_return = sum(mu[i] for i in selected_indices)
    portfolio_variance = sum(
        sigma[i][j] for i in selected_indices for j in selected_indices
    )
    portfolio_std = np.sqrt(portfolio_variance)
    return portfolio_return / portfolio_std if portfolio_std != 0 else -np.inf


def calculate_profit(future_returns, selected_assets):
    """
    Oblicza średni zysk z portfela zbudowanego na podstawie wybranych aktywów.

    :param future_returns: DataFrame zwrotów z kolejnego okna po optymalizacji
    :param selected_assets: np.array z 0 i 1 określającymi wybrane aktywa
    :return: średni zysk portfela w przyszłości
    """
    if future_returns.empty:
        return np.nan

    selected_indices = np.where(selected_assets == 1)[0]
    if len(selected_indices) == 0:
        return 0.0

    selected_returns = future_returns.iloc[:, selected_indices]
    portfolio_returns = selected_returns.mean(axis=1)
    return portfolio_returns.mean()