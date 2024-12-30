# %%

import os
import pandas as pd

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

deck_full = pd.read_csv("de-en/deck.csv", sep="|", index_col=0)
deck_full.index = deck_full.index.map(lambda x: x if x % 1 != 0 else f"{int(x)}")
deck_part_1 = deck_full.loc[deck_full["frequency_rank"] <= 3000]
deck_part_2 = deck_full.loc[deck_full["frequency_rank"] > 3000]
deck_part_1.to_csv("de-en/deck-1.csv", sep="|")
deck_part_2.to_csv("de-en/deck-2.csv", sep="|")
