import pandas as pd
df = pd.read_csv('val.csv')
filtered_df = df[df['label'].isin([0, 1])]
filtered_df.to_csv('filtered_val.csv', index=False)
print(f"Original file had {len(df)} rows")
print(f"Filtered file has {len(filtered_df)} rows")
print("Saved filtered data to 'filtered_val.csv'")
