import pandas as pd
val_file = r"C:\Users\STIC-11\Desktop\sk1\csv\test.csv"
val_df = pd.read_csv(val_file)
filtered_val_df = val_df[val_df['level'] == 4]
filtered_val_df.to_csv("test_proliferate.csv", index=False)
print("Filtered CSV file saved as 'test_moderate.csv'")
print(f"Selected {len(filtered_val_df)} images with level value 2")
