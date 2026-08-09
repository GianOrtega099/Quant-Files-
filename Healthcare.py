import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_healthcare_policy_volatility():
    print("Initializing Healthcare Policy Volatility Engine...\n")
    
    # 1. Define sub-sector tickers representing different policy sensitivities
    # UNH/CI = Insurers, PFE/MRK = Pharma, HCA = Hospital Systems
    tickers = {
        "UNH": "Managed Care (Insurers)",
        "CI":  "Managed Care (Insurers)",
        "PFE": "Pharmaceuticals",
        "MRK": "Pharmaceuticals",
        "HCA": "Hospital Providers"
    }
    
    ticker_list = list(tickers.keys())
    
    # 2. Historical window capturing recent major administration & policy shifts
    start_date = "2024-01-01"
    end_date = "2026-05-15"
    
    print(f"Downloading historical pricing arrays from yfinance ({start_date} to {end_date})...")
    raw_data = yf.download(ticker_list, start=start_date, end=end_date)
    
    # Isolate Close prices and drop any missing rows
    df_close = raw_data['Close'].dropna()
    
    # 3. Calculate Log Returns via NumPy for statistical normality
    # Log Return = ln(Price_t / Price_t-1)
    df_returns = pd.DataFrame()
    for ticker in ticker_list:
        df_returns[ticker] = np.log(df_close[ticker] / df_close[ticker].shift(1))
    
    df_returns = df_returns.dropna()
    
    # 4. Map Critical Administration Policy Event Dates
    # We define key windows where major healthcare bills or executive actions dropped
    policy_events = {
        "2025-06-15": "OBBBA Reconciliation Bill Enacted (Medicaid/ACA Shift)",
        "2026-01-01": "Medicare Part D Drug Price Caps Take Effect",
        "2026-01-15": "White House Releases 'The Great Healthcare Plan'",
        "2026-05-15": "Congress Passes 'The Great Healthcare Plan'",
    }

    # Estimate historical volatility on log returns
    df_vol = df_returns.std() * np.sqrt(252)

    print("\n=========================================================")
    print(" STATISTICAL RESULTS: HISTORICAL SECTOR BASELINE VOLATILITY ")
    print("=========================================================")
    # Establish baseline risk profiles across the entire historical period
    for ticker in ticker_list:
        mean_vol = df_vol[ticker].mean() * 100
        max_vol = df_vol[ticker].max() * 100
        print(f"Ticker: {ticker:<4} | Sector: {tickers[ticker]:<24} | Base Vol: {mean_vol:6.2f}% | Peak Vol: {max_vol:6.2f}%")
        
    print("\n=========================================================")
    print(" EVENT STUDY ANALYSIS: IMPACT OF ADMINISTRATIVE POLICIES  ")
    print("=========================================================")
    
    # Run the event window loop
    # We analyze price fluctuations 5 days prior to an announcement and 10 days post-announcement
    pre_event_days = 5
    post_event_days = 10
    
    for event_date_str, event_name in policy_events.items():
        event_date = pd.to_datetime(event_date_str)        
        
        # Locate closest actual trading day in our pandas index
        if event_date not in df_returns.index:
            # Find the nearest subsequent trading day if event fell on a weekend
            available_dates = df_returns.index[df_returns.index >= event_date]
            if len(available_dates) == 0:
                continue
            event_date = available_dates[0]
            
        # Extract positional index for window slicing
        idx = df_returns.index.get_loc(event_date)
        
        # Ensure windows don't run out of bounds of the dataset
        if idx - pre_event_days < 0 or idx + post_event_days >= len(df_returns):
            continue
            
        print(f"\n[EVENT]: {event_name}")
        print(f"Target Cluster Window: {df_returns.index[idx-pre_event_days].strftime('%Y-%m-%d')} to {df_returns.index[idx+post_event_days].strftime('%Y-%m-%d')}")
        print("-" * 75)
        
        for ticker in ticker_list:
            # Isolate the returns sequence during the policy shift window
            window_returns = df_returns[ticker].iloc[idx - pre_event_days : idx + post_event_days]
            
            # Compute total Cumulative Return over the policy shock window via exponential sum
            cumulative_return = (np.exp(window_returns.sum()) - 1) * 100
            
            # Compute localized volatility during the event window
            event_window_vol = window_returns.std() * np.sqrt(252) * 100
            
            # Compute historical average volatility for context
            historical_avg_vol = df_vol[ticker].mean() * 100
            
            # Volatility Expansion Ratio (Did policy cause abnormal fluctuation frequency?)
            vol_multiplier = event_window_vol / historical_avg_vol
            
            print(f"  Ticker: {ticker:<4} | Cum. Window Return: {cumulative_return:+6.2f}% | Event Window Vol: {event_window_vol:6.2f}% (x{vol_multiplier:.2f} Baseline)")

if __name__ == "__main__":
    analyze_healthcare_policy_volatility()