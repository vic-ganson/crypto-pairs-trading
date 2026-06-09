import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Computes certain performance metrics given daily returns (252 = trading days / year)
def compute_sharpe(returns, periods=252):
    mean = returns.mean() * periods
    vol = returns.std() * np.sqrt(periods)
    return mean / vol if vol != 0 else 0.0

def compute_annualized_return(returns, periods=252):
    return returns.mean() * periods

def compute_annualized_volatility(returns, periods=252):
    return returns.std() * np.sqrt(periods)


def compute_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def compute_max_drawdown(returns):
    return compute_drawdown(returns).min()

# Most days spent below previous peak
def compute_max_drawdown_duration(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()

    # True on days we are below the peak
    below = cumulative < running_max
    max_duration = 0
    current_duration = 0
    for is_below in below:
        if is_below:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration

# For the benchmark of this information ratio, I used buying and holding Bitcoin. Excess return per unit of risk
def compute_information_ratio(strategy_returns, benchmark_returns, window=252):
    combined = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Fix: unstack the MultiIndex before indexing
    rolling_corr = combined.rolling(window).corr().unstack()['strategy']['benchmark']
    rolling_vol = combined.rolling(window).std()

    beta = (rolling_corr * rolling_vol['strategy']) / rolling_vol['benchmark']
    residuals = combined['strategy'] - beta * combined['benchmark']

    ir = residuals.mean() / residuals.std() * np.sqrt(252)
    return ir, residuals

# Average number of days a pos. is held
def compute_holding_period(turnover):
    avg_turnover = turnover.mean()
    if avg_turnover != 0:
      return 2 / avg_turnover
    else:
      return np.inf

# Summarizes stats
def compute_stats(net_returns, gross_returns, turnover, benchmark_returns=None, tcost_bps=20):
    stats = {}
    stats['Annualized Return']     = compute_annualized_return(net_returns)
    stats['Annualized Volatility'] = compute_annualized_volatility(net_returns)
    stats['Sharpe Ratio']          = compute_sharpe(net_returns)
    stats['Max Drawdown']          = compute_max_drawdown(net_returns)
    stats['Max Drawdown Duration'] = compute_max_drawdown_duration(net_returns)
    stats['Avg Daily Turnover']    = turnover.mean()
    stats['Holding Period (days)'] = compute_holding_period(turnover)
    stats['Avg Daily Tcost']       = (turnover * tcost_bps * 1e-4).mean()

    if benchmark_returns is not None:
        ir, _ = compute_information_ratio(net_returns, benchmark_returns)
        stats['Information Ratio'] = ir

    stats_df = pd.Series(stats).round(4)
    print(stats_df.to_string())
    return stats_df

# Plots cumultative returns of strategy and benchmark (optionally)
def plot_cumulative_returns(net_returns, benchmark_returns=None):
    cumulative = (1 + net_returns).cumprod()

    plt.figure(figsize=(12, 5))
    plt.plot(cumulative.index, cumulative, label='Strategy', linewidth=1.5)

    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        plt.plot(bench_cumulative.index, bench_cumulative,
                 label='Buy & Hold BTC', linewidth=1.5, linestyle='--')

    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plots drawdown series over time
def plot_drawdown(returns, title="Drawdown"):
    drawdown = compute_drawdown(returns)

    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.4)
    plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plots Sharpe ratio with 1-year rolling windows. Will tell us if performance is consistent
def plot_rolling_sharpe(returns, window=252, title="Rolling Sharpe Ratio"):
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_vol

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, linewidth=1.5)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
