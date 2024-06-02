# %%

import os

os.chdir(f'{os.environ["ROOT_DIR"]}/word-combinations/de')

# %%

import pandas as pd

one_meaning_one_line = pd.read_csv(
    "data/processed/de-en-one-meaning-one-line.csv", sep="|", header=None
)
one_meaning_one_line

# %%
word_combinations = pd.read_csv(
    "data/processed/de-en-word-combinations.csv", sep="|", header=None
)

# %%
combined = pd.concat(
    [word_combinations, one_meaning_one_line.loc[:, [1, 2, 3, 4, 5, 0]]], axis=1
)

# %%
combined.columns = pd.Index(
    [
        "word_combination_de",
        "word_combination_en",
        "word_de",
        "kind",
        "word_en",
        "sentence_de",
        "sentence_en",
        "frequency_rank",
    ]
)


# %%

combined[
    [
        "kind",
        "word_de",
        # "word_combination_de",
        "word_en",
        # "word_combination_en",
        # "sentence_de",
        # "sentence_en",
        # "frequency_rank",
    ]
].to_csv(path_or_buf="data/processed/de-en-words.csv", sep="|", index=False)
