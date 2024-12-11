import pandas as pd

# File path to the CSV file
csv_file = "data/airlines.csv"  # Replace with your actual CSV file path

# Desired column types
column_types = {"Flight": int, "DayOfWeek": int, "Time": int, "Length": int, "Delay": int}

# Load the CSV file
df = pd.read_csv(csv_file)

# Convert specified columns to the desired types
for column, dtype in column_types.items():
    if column in df.columns:
        df[column] = df[column].astype(dtype)

# Save the updated DataFrame back to a CSV file
output_file = "data/airlines.csv"
df.to_csv(output_file, index=False)

print(f"Columns converted and saved to {output_file}.")
