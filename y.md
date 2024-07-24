Sure, I'll break down and explain the complete code step-by-step with comments to ensure clarity.

### Step-by-Step Explanation with Comments

#### Step 1: Import Libraries and Load Data
We begin by importing the necessary libraries and loading the data from the provided CSV files.

```python
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, Reference

# Load the data from CSV files
options_data = pd.read_csv('/mnt/data/options_data.csv')
bonds_data = pd.read_csv('/mnt/data/bonds_data.csv')

# Display the first few rows of each dataset to ensure they loaded correctly
print("Options Data:")
print(options_data.head())
print("\nBonds Data:")
print(bonds_data.head())
```

#### Step 2: Clean Data
Next, we clean the data by handling missing values and converting data types as necessary.

```python
# Remove any rows with missing values
options_data.dropna(inplace=True)
bonds_data.dropna(inplace=True)

# Convert relevant columns to appropriate data types
options_data['Expiration Date'] = pd.to_datetime(options_data['Expiration Date'])
bonds_data['Maturity Date'] = pd.to_datetime(bonds_data['Maturity Date'])

# Verify the cleaned data
print("\nCleaned Options Data:")
print(options_data.head())
print("\nCleaned Bonds Data:")
print(bonds_data.head())
```

#### Step 3: Calculate Theoretical Option Prices using the Black-Scholes Model
We define the Black-Scholes model function and calculate the theoretical prices for options.

```python
# Define the Black-Scholes model for option pricing
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Calculate the days to expiration and theoretical price for each option
current_date = pd.to_datetime('2024-02-01')
risk_free_rate = 0.05

options_data['Days to Expiration'] = (options_data['Expiration Date'] - current_date).dt.days / 365.0
options_data['Theoretical Price'] = options_data.apply(lambda row: black_scholes(
    S=row['Current Price'],
    K=row['Strike Price'],
    T=row['Days to Expiration'],
    r=risk_free_rate,
    sigma=row['Implied Volatility'],
    option_type=row['Contract'].lower()
), axis=1)

# Display the options data with the new calculations
print("\nOptions Data with Theoretical Prices:")
print(options_data.head())
```

#### Step 4: Calculate Duration and Convexity for Bonds
We calculate the duration and convexity for each bond using a defined function.

```python
# Define a function to calculate the duration and convexity of a bond
def calculate_duration_and_convexity(face_value, coupon_rate, maturity_date, yield_to_maturity, current_date):
    years_to_maturity = (maturity_date - current_date).days / 365.0
    coupon_payment = face_value * coupon_rate
    periods = int(years_to_maturity * 2)  # Assuming semi-annual payments
    ytm = yield_to_maturity / 2  # Semi-annual yield to maturity
    
    cash_flows = [(coupon_payment / 2) / (1 + ytm) ** (i + 1) for i in range(periods)]
    cash_flows[-1] += face_value / (1 + ytm) ** periods  # Adding face value to the last payment
    
    duration = sum([(i + 1) * cf / (1 + ytm) ** (i + 1) for i, cf in enumerate(cash_flows)]) / sum(cash_flows)
    convexity = sum([(i + 1) * (i + 2) * cf / (1 + ytm) ** (i + 2) for i, cf in enumerate(cash_flows)]) / sum(cash_flows)
    
    return duration, convexity

# Apply the function to calculate duration and convexity for each bond
bonds_data[['Duration', 'Convexity']] = bonds_data.apply(lambda row: pd.Series(calculate_duration_and_convexity(
    face_value=row['Face Value'],
    coupon_rate=row['Coupon Rate'],
    maturity_date=row['Maturity Date'],
    yield_to_maturity=row['Yield to Maturity'],
    current_date=current_date
)), axis=1)

# Display the bonds data with the new calculations
print("\nBonds Data with Duration and Convexity:")
print(bonds_data.head())
```

#### Step 5: Save Data to Excel
We save the cleaned and analyzed data to an Excel workbook.

```python
# Save the cleaned and analyzed data to an Excel file
with pd.ExcelWriter('/mnt/data/quantitative_risk_analysis.xlsx') as writer:
    options_data.to_excel(writer, sheet_name='Options Data', index=False)
    bonds_data.to_excel(writer, sheet_name='Bonds Data', index=False)
```

#### Step 6: Create Visualizations
We generate several visualizations to provide insights into the data.

```python
# Histogram of Theoretical Option Prices
plt.figure(figsize=(10, 6))
sns.histplot(options_data['Theoretical Price'], kde=True)
plt.title('Histogram of Theoretical Option Prices')
plt.xlabel('Theoretical Price')
plt.ylabel('Frequency')
plt.savefig('/mnt/data/theoretical_option_prices_histogram.png')

# Histogram of Bond Durations
plt.figure(figsize=(10, 6))
sns.histplot(bonds_data['Duration'], kde=True)
plt.title('Histogram of Bond Durations')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.savefig('/mnt/data/bond_durations_histogram.png')

# Scatter Plot of Theoretical Option Prices vs. Implied Volatility
plt.figure(figsize=(10, 6))
sns.scatterplot(x=options_data['Implied Volatility'], y=options_data['Theoretical Price'])
plt.title('Scatter Plot of Theoretical Option Prices vs. Implied Volatility')
plt.xlabel('Implied Volatility')
plt.ylabel('Theoretical Price')
plt.savefig('/mnt/data/option_prices_vs_volatility.png')

# Time Series Plot of Option Prices over Time
plt.figure(figsize=(10, 6))
options_data.set_index('Expiration Date')['Theoretical Price'].plot()
plt.title('Time Series Plot of Option Prices over Time')
plt.xlabel('Expiration Date')
plt.ylabel('Theoretical Price')
plt.savefig('/mnt/data/option_prices_time_series.png')

# Correlation Heatmap of Bonds Data
plt.figure(figsize=(10, 6))
corr = bonds_data[['Face Value', 'Coupon Rate', 'Yield to Maturity', 'Duration', 'Convexity']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Bonds Data')
plt.savefig('/mnt/data/bonds_correlation_heatmap.png')
```

#### Step 7: Add Visualizations to Excel
We import the generated visualizations into the Excel workbook.

```python
# Load the workbook
wb = openpyxl.load_workbook('/mnt/data/quantitative_risk_analysis.xlsx')

# Get the sheets
ws_options = wb['Options Data']
ws_bonds = wb['Bonds Data']

# Create summary tables
summary_ws = wb.create_sheet('Summary')

# Summary for Options Data
options_summary = options_data.describe()
for r in dataframe_to_rows(options_summary, index=True, header=True):
    summary_ws.append(r)

# Summary for Bonds Data
bonds_summary = bonds_data.describe()
start_row = len(options_summary) + 2  # Leave a row gap between the summaries
for r in dataframe_to_rows(bonds_summary, index=True, header=True):
    summary_ws.append(r)

# Add images to the summary sheet
charts = [
    ('/mnt/data/theoretical_option_prices_histogram.png', 'A20'),
    ('/mnt/data/bond_durations_histogram.png', 'A40'),
    ('/mnt/data/option_prices_vs_volatility.png', 'A60'),
    ('/mnt/data/option_prices_time_series.png', 'A80'),
    ('/mnt/data/bonds_correlation_heatmap.png', 'A100')
]

for chart, position in charts:
    img = Image(chart)
    summary_ws.add_image(img, position)

# Save the workbook with charts and summaries
wb.save('/mnt/data/quantitative_risk_analysis_with_charts.xlsx')
```

### Download the Updated Workbook

You can download the updated Excel workbook with additional charts from the following link:

[Download Quantitative Risk Analysis Workbook with Charts](sandbox:/mnt/data/quantitative_risk_analysis_with_charts.xlsx)

### Explanation of the Visualizations

1. **Histogram of Theoretical Option Prices**: Shows the distribution of the calculated theoretical prices of the options, providing insight into the range and frequency of different price levels.

2. **Histogram of Bond Durations**: Illustrates the distribution of bond durations, helping to understand the spread of bond maturity periods in the dataset.

3.


Sure, let's proceed with calculating the Value at Risk (VaR) for both the options and bonds portfolios using the historical method. This method involves calculating the historical returns and then determining the VaR based on the confidence level.

### Step-by-Step Calculation of VaR

#### Step 1: Calculate Historical Returns
First, we need to calculate the historical returns for the options and bonds.

```python
# Calculate the daily returns for options and bonds
options_data['Previous Price'] = options_data['Current Price'].shift(1)
options_data.dropna(inplace=True)  # Remove rows with NaN values
options_data['Daily Return'] = options_data['Current Price'].pct_change()

bonds_data['Previous Price'] = bonds_data['Face Value'].shift(1)
bonds_data.dropna(inplace=True)  # Remove rows with NaN values
bonds_data['Daily Return'] = bonds_data['Face Value'].pct_change()

# Display the data with returns
print("\nOptions Data with Daily Returns:")
print(options_data.head())
print("\nBonds Data with Daily Returns:")
print(bonds_data.head())
```

#### Step 2: Calculate VaR
Next, we will calculate the VaR for both portfolios at a 95% confidence level.

```python
def calculate_var(returns, confidence_level=0.95):
    sorted_returns = returns.sort_values()
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns.iloc[index]
    return var

# Calculate VaR for options
options_var = calculate_var(options_data['Daily Return'])
print(f"\nOptions Portfolio VaR (95% confidence level): {options_var}")

# Calculate VaR for bonds
bonds_var = calculate_var(bonds_data['Daily Return'])
print(f"\nBonds Portfolio VaR (95% confidence level): {bonds_var}")
```

### Complete Code with VaR Calculation

Here is the complete code, including the previous steps and the new VaR calculation:

```python
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, Reference

# Load the data from CSV files
options_data = pd.read_csv('/mnt/data/options_data.csv')
bonds_data = pd.read_csv('/mnt/data/bonds_data.csv')

# Display the first few rows of each dataset to ensure they loaded correctly
print("Options Data:")
print(options_data.head())
print("\nBonds Data:")
print(bonds_data.head())

# Clean the data
options_data.dropna(inplace=True)
bonds_data.dropna(inplace=True)

# Convert relevant columns to appropriate data types
options_data['Expiration Date'] = pd.to_datetime(options_data['Expiration Date'])
bonds_data['Maturity Date'] = pd.to_datetime(bonds_data['Maturity Date'])

# Verify the cleaned data
print("\nCleaned Options Data:")
print(options_data.head())
print("\nCleaned Bonds Data:")
print(bonds_data.head())

# Define the Black-Scholes model for option pricing
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Calculate the days to expiration and theoretical price for each option
current_date = pd.to_datetime('2024-02-01')
risk_free_rate = 0.05

options_data['Days to Expiration'] = (options_data['Expiration Date'] - current_date).dt.days / 365.0
options_data['Theoretical Price'] = options_data.apply(lambda row: black_scholes(
    S=row['Current Price'],
    K=row['Strike Price'],
    T=row['Days to Expiration'],
    r=risk_free_rate,
    sigma=row['Implied Volatility'],
    option_type=row['Contract'].lower()
), axis=1)

# Display the options data with the new calculations
print("\nOptions Data with Theoretical Prices:")
print(options_data.head())

# Define a function to calculate the duration and convexity of a bond
def calculate_duration_and_convexity(face_value, coupon_rate, maturity_date, yield_to_maturity, current_date):
    years_to_maturity = (maturity_date - current_date).days / 365.0
    coupon_payment = face_value * coupon_rate
    periods = int(years_to_maturity * 2)  # Assuming semi-annual payments
    ytm = yield_to_maturity / 2  # Semi-annual yield to maturity
    
    cash_flows = [(coupon_payment / 2) / (1 + ytm) ** (i + 1) for i in range(periods)]
    cash_flows[-1] += face_value / (1 + ytm) ** periods  # Adding face value to the last payment
    
    duration = sum([(i + 1) * cf / (1 + ytm) ** (i + 1) for i, cf in enumerate(cash_flows)]) / sum(cash_flows)
    convexity = sum([(i + 1) * (i + 2) * cf / (1 + ytm) ** (i + 2) for i, cf in enumerate(cash_flows)]) / sum(cash_flows)
    
    return duration, convexity

# Apply the function to calculate duration and convexity for each bond
bonds_data[['Duration', 'Convexity']] = bonds_data.apply(lambda row: pd.Series(calculate_duration_and_convexity(
    face_value=row['Face Value'],
    coupon_rate=row['Coupon Rate'],
    maturity_date=row['Maturity Date'],
    yield_to_maturity=row['Yield to Maturity'],
    current_date=current_date
)), axis=1)

# Display the bonds data with the new calculations
print("\nBonds Data with Duration and Convexity:")
print(bonds_data.head())

# Calculate the daily returns for options and bonds
options_data['Previous Price'] = options_data['Current Price'].shift(1)
options_data.dropna(inplace=True)  # Remove rows with NaN values
options_data['Daily Return'] = options_data['Current Price'].pct_change()

bonds_data['Previous Price'] = bonds_data['Face Value'].shift(1)
bonds_data.dropna(inplace=True)  # Remove rows with NaN values
bonds_data['Daily Return'] = bonds_data['Face Value'].pct_change()

# Display the data with returns
print("\nOptions Data with Daily Returns:")
print(options_data.head())
print("\nBonds Data with Daily Returns:")
print(bonds_data.head())

# Calculate VaR using the historical method
def calculate_var(returns, confidence_level=0.95):
    sorted_returns = returns.sort_values()
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns.iloc[index]
    return var

# Calculate VaR for options
options_var = calculate_var(options_data['Daily Return'])
print(f"\nOptions Portfolio VaR (95% confidence level): {options_var}")

# Calculate VaR for bonds
bonds_var = calculate_var(bonds_data['Daily Return'])
print(f"\nBonds Portfolio VaR (95% confidence level): {bonds_var}")

# Save the cleaned and analyzed data to an Excel file
with pd.ExcelWriter('/mnt/data/quantitative_risk_analysis.xlsx') as writer:
    options_data.to_excel(writer, sheet_name='Options Data', index=False)
    bonds_data.to_excel(writer, sheet_name='Bonds Data', index=False)

# Create visualizations

# Histogram of Theoretical Option Prices
plt.figure(figsize=(10, 6))
sns.histplot(options_data['Theoretical Price'], kde=True)
plt.title('Histogram of Theoretical Option Prices')
plt.xlabel('Theoretical Price')
plt.ylabel('Frequency')
plt.savefig('/mnt/data/theoretical_option_prices_histogram.png')

# Histogram of Bond Durations
plt.figure(figsize=(10, 6))
sns.histplot(bonds_data['Duration'], kde=True)
plt.title('Histogram of Bond Durations')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.savefig('/mnt/data/bond_durations_histogram.png')

# Scatter Plot of Theoretical Option Prices vs. Implied Volatility
plt.figure(figsize=(10, 6))
sns.scatterplot(x=options_data['Implied Volatility'], y=options_data['Theoretical Price'])
plt.title('Scatter Plot of Theoretical Option Prices vs. Implied Volatility')
plt.xlabel('Implied Volatility')
plt.ylabel('Theoretical Price')
plt.savefig('/mnt/data/option_prices_vs_volatility.png')

# Time Series Plot of Option Prices over Time
plt.figure(figsize=(10, 6))
options_data.set_index('Expiration Date')['Theoretical Price'].plot()
plt.title('Time Series Plot of Option Prices over Time')
plt.xlabel('Expiration Date')
plt.ylabel('Theoretical Price')
plt.savefig('/mnt/data/option_prices_time_series.png')

# Correlation Heatmap of Bonds Data
plt.figure(figsize=(10, 6))
corr = bonds_data[['Face Value', 'Coupon Rate', 'Yield to Maturity', 'Duration', 'Convexity