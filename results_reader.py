import pandas as pd

df = pd.read_json('outputs/math_example_3signals_newthresholds_llama_run2.json')

resets = 0

for index, row in df.iterrows():
  resets += row["resets"].count(1)

print("Total # of resets:", resets)