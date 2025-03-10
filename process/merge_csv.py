import pandas as pd
df1 = pd.read_csv(r"C:\Users\STIC-11\Desktop\sk1\train_square_3.csv")
df2 = pd.read_csv(r"C:\Users\STIC-11\Desktop\sk1\csv\test_proliferate.csv") 
merged_df = pd.concat([df1, df2]).drop_duplicates()
merged_df = merged_df.sort_values('image')
merged_df.to_csv('train_square_4.csv', index=False)