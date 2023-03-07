#%%
# should = {x for x in range(1, 20886)}
s = None
with open("data/check/loaded.csv") as loaded:
    s = {x for x in range(1, 20886)} - {
        int(x.split(" ")[1][:-1]) for x in loaded.read().split("\n")
    }
# %%
print(s)
#%%
t = None
with open("data/check/not_loaded.csv", "a+", encoding="utf8") as not_loaded:
  not_loaded.write("|".join([f"header_{i}" for i in range(12)]) + "\n")
  with open("data/check/en-ru.csv", "r", encoding="utf8") as enru:
        for p, t in enumerate(enru.read().split("\n"), 1):
            if p in s:
                not_loaded.write(f"{t}\n")
#%%
import pandas as pd

df = pd.read_csv("data/check/not_loaded.csv", sep="@", encoding="utf8")
df
#%%
df
# %%
df[["header_8","header_9","header_10","header_11"]]
# %%
