import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Read the data from a CSV file
file_path = 'C:\\Users\\USER\\Documents\\Research\\Development\\sentiment_data.csv'  # Replace with the path to your new CSV file
df = pd.read_csv(file_path)

# Convert 'Year' and 'Month' to datetime if needed
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

# Define the sets
set1 = df.loc[(df['Date'] >= '2021-04-01') & (df['Date'] <= '2021-09-30')]
set2 = df.loc[(df['Date'] >= '2021-10-01') & (df['Date'] <= '2022-03-31')]
set3 = df.loc[(df['Date'] >= '2022-04-01') & (df['Date'] <= '2022-09-30')]
set4 = df.loc[(df['Date'] >= '2022-10-01') & (df['Date'] <= '2023-03-31')]

# Create 'Set' column in the DataFrame
df['Set'] = pd.cut(df['Date'], bins=[set1['Date'].min(), set2['Date'].max(), set3['Date'].max(), set4['Date'].max()], labels=['Set1', 'Set2', 'Set3'])

# Perform ANOVA for each variable
variables = [
    'ConfidenceofPositiveSentiment',
    'ConfidenceofNeutralSentiment',
    'ConfidenceofNegativeSentiment'
]

results = {}

for variable in variables:
    formula = f"{variable} ~ C(Set)"
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model)
    p_value = anova_table['PR(>F)']['C(Set)']

    results[variable] = {
        'F-statistic': anova_table['F']['C(Set)'],
        'P-value': p_value
    }

# Display results
for variable, values in results.items():
    print(f"\nResults for {variable}:\n")
    print(f"F-statistic: {values['F-statistic']}")
    print(f"P-value: {values['P-value']}")
    print("--------------------------------------------------------")
