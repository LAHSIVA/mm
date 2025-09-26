import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_moving_averages(data, window_size=4):
  
    n = len(data)
    
    #  Calculate the 4-Quarter Moving Total
    moving_total = []
    for i in range(n - window_size + 1):
        total = sum(data[i : i + window_size])
        moving_total.append(total)
        
    # Calculate the 4-Quarter Moving Average
    moving_average = [total / window_size for total in moving_total]
    
    #  Center the Moving Average
    centered_moving_avg = []
    for i in range(len(moving_average) - 1):
        centered_avg = (moving_average[i] + moving_average[i+1]) / 2
        centered_moving_avg.append(centered_avg)
        
    return centered_moving_avg, moving_total, moving_average

def calculate_seasonal_indices(actual_data, centered_moving_avg):
    # Calculate the percentage of actual value to moving average
    percentage_actual_to_ma = []
    for i in range(len(centered_moving_avg)):
        ratio = (actual_data[i+2] / centered_moving_avg[i]) * 100
        percentage_actual_to_ma.append(round(ratio, 3))

    # Group the percentages by quarter to find the average effect
    quarters = [[], [], [], []] 
  
    for i, perc in enumerate(percentage_actual_to_ma):
        quarter_index = (i + 2) % 4
        if quarter_index == 2: quarters[0].append(perc) 
        elif quarter_index == 3: quarters[1].append(perc) 
        elif quarter_index == 0: quarters[2].append(perc) 
        elif quarter_index == 1: quarters[3].append(perc) 

    # Calculate the Modified Mean for each quarter
    
    modified_mean = []
    for q_data in quarters:
        if len(q_data) > 2:
            mean = (sum(q_data) - max(q_data) - min(q_data)) / (len(q_data) - 2)
        else: 
            mean = sum(q_data) / len(q_data) if q_data else 0
        modified_mean.append(round(mean, 3))

    # Adjust the indices to sum to 400
   
    total_modified_mean = sum(modified_mean)
    adjustment_factor = 400 / total_modified_mean if total_modified_mean != 0 else 1
    
    seasonal_indices = [round(mean * adjustment_factor, 3) for mean in modified_mean]
    
    return seasonal_indices, percentage_actual_to_ma, quarters, modified_mean, adjustment_factor

def calculate_trend_and_cyclic(deseasonalized_data):
  
    n = len(deseasonalized_data)
    
    X = [i - (n - 1) / 2 for i in range(n)]

    XY = [x * y for x, y in zip(X, deseasonalized_data)]
    X2 = [x**2 for x in X]
    
    b = sum(XY) / sum(X2) if sum(X2) != 0 else 0
    a = sum(deseasonalized_data) / n
    
    trend_values = [a + b * x for x in X]
    
    cyclic_variation = [(y / y_pred) * 100 for y, y_pred in zip(deseasonalized_data, trend_values)]
    
    return a, b, trend_values, cyclic_variation, X, XY, X2

def main():
 
    years = [1992, 1993, 1994, 1995]
    actual_values = [293, 246, 231, 282, 301, 252, 227, 291, 304, 259, 239, 296, 306, 265, 240, 300]
    n = len(actual_values)
    
    # Calculate Moving Averages
    centered_ma, moving_total, moving_avg = calculate_moving_averages(actual_values)
    
    # Calculate Seasonal Component
    seasonal_indices, percentage_ratio, quarters_grouped, modified_means, adj_factor = \
        calculate_seasonal_indices(actual_values, centered_ma)

    # Deseasonalize the Data
    deseasonalized_data = []
    for i in range(n):
        seasonal_factor = seasonal_indices[i % 4] / 100
        deseasonalized_data.append(round(actual_values[i] / seasonal_factor, 3))
        
    # Calculate Trend and Cyclic Components
    a, b, trend_line, cyclic_var, X, XY, X2 = calculate_trend_and_cyclic(deseasonalized_data)

    # Display Results in Tables
    display_results(
        years, actual_values, moving_total, moving_avg, centered_ma, percentage_ratio,
        quarters_grouped, modified_means, adj_factor, seasonal_indices,
        deseasonalized_data, a, b, trend_line, cyclic_var, X, XY, X2
    )
    
    # Plot the Data
    plot_data(years, actual_values, deseasonalized_data, trend_line, centered_ma)

def display_results(years, yVal, mov_tot, mov_avg, cent_mov_avg, perc_ratio, q_grouped,
                    mod_means, adj_factor, seas_indices, deseason_data, a, b, y_pred,
                    cyc_var, X, XY, X2):
   
    year_display = [y for y in years for _ in range(4)]
    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4'] * len(years)

    # Moving Averages and Ratios
    table1 = pd.DataFrame({
        "Year": year_display,
        "Quarter": quarter_labels,
        "Actual Value": yVal,
        "Moving Total": [''] * 2 + mov_tot + [''] * 1,
        "Moving Average": [''] * 2 + mov_avg + [''] * 1,
        "Centered Moving Avg": [''] * 2 + cent_mov_avg + [''] * 2,
        "Ratio (Actual/CMA)": [''] * 2 + perc_ratio + [''] * 2
    })
    print("Table 1: Calculation of Moving Averages and Ratios")
    print(table1.to_string())

    # Calculating Seasonal Indices
    table2_data = {}
    for i in range(4):
        
        col_data = q_grouped[i] + [''] * (len(years) - len(q_grouped[i]))
        table2_data[f"Quarter {i+1}"] = col_data
    
    table2 = pd.DataFrame(table2_data, index=years)
    table2.loc['Modified Sum'] = [round(sum(q) - max(q) - min(q), 3) if len(q)>2 else sum(q) for q in q_grouped]
    table2.loc['Modified Mean'] = mod_means
    print("\n--- Table 2: Grouping Ratios and Calculating Modified Mean ---")
    print(table2)

    print(f"\nAdjustment Factor: {adj_factor:.4f}")
    table3 = pd.DataFrame({
        "Quarter": ['Q1', 'Q2', 'Q3', 'Q4'],
        "Modified Mean": mod_means,
        "Seasonal Index": seas_indices
    })
    print("\n--- Table 3: Final Seasonal Indices ---")
    print(table3)
    print(f"Sum of seasonal indices: {sum(seas_indices):.1f}")
    
    # Deseasonalized Data
    table4 = pd.DataFrame({
        "Year": year_display,
        "Quarter": quarter_labels,
        "Actual Value": yVal,
        "Seasonal Index": seas_indices * len(years),
        "Deseasonalized Data": deseason_data
    })
    print("\n--- Table 4: Deseasonalized Time Series ---")
    print(table4.to_string())

    # Trend Calculation
    table5 = pd.DataFrame({
        "Year": year_display,
        "Quarter": quarter_labels,
        "Deseasonalized (Y)": deseason_data,
        "Time (X)": X,
        "XY": XY,
        "X^2": X2
    })
    print("\n--- Table 5: Trend Component Calculation ---")
    print(table5.to_string())
    print(f"\nTrend Line Equation: Y = {a:.3f} + {b:.3f}X")

    # Cyclic Variation
    table6 = pd.DataFrame({
        "Year": year_display,
        "Quarter": quarter_labels,
        "Deseasonalized (Y)": deseason_data,
        "Trend Value (Y_pred)": [round(y, 3) for y in y_pred],
        "Cyclic Var (%)": [round(c, 3) for c in cyc_var]
    })
    print("\n--- Table 6: Cyclic Variation Calculation ---")
    print(table6.to_string())

def plot_data(years, actual, deseasonalized, trend, centered_ma):
    
    labels = [f"{yr} {q}" for yr in years for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    
    plt.figure(figsize=(14, 8))
    
   
    plt.plot(labels, actual, marker='o', linestyle='-', label="Actual Data")
    plt.plot(labels, deseasonalized, marker='s', linestyle='--', label="Deseasonalized Data")
    plt.plot(labels, trend, marker='x', linestyle=':', color='red', label="Trend Line")
    
   
    padded_ma = [np.nan, np.nan] + centered_ma + [np.nan, np.nan]
    plt.plot(labels, padded_ma, marker='d', linestyle='-.', color='green', label="Centered Moving Average")

    plt.title("Time Series Decomposition Analysis", fontsize=16)
    plt.xlabel("Year and Quarter", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    main()
