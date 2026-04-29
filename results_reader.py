import pandas as pd

df = pd.read_json('outputs/math_example_ergo_selectthreshold_gpt41mini_run2.json')

# print(df.shape[0])
# print(df.columns)

resets = 0

for index, row in df.iterrows():
#   print(row['item_id'])
#   print(f"{row["resets"].count(1)} out of {len(row["resets"])}")
  resets += row["resets"].count(1)

print("Total # of resets:", resets)