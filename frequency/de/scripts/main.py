# %%

import os
import pandas as pd

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

# %%


def partition_deck_data():
    os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

    deck_full = pd.read_csv("de-en/deck.csv", sep="|", index_col=0)
    deck_full.index = deck_full.index.map(lambda x: x if x % 1 != 0 else f"{int(x)}")
    deck_part_1 = deck_full.loc[deck_full["frequency_rank"] <= 3000]
    deck_part_2 = deck_full.loc[deck_full["frequency_rank"] > 3000]
    deck_part_1.to_csv("de-en/deck-1.csv", sep="|")
    deck_part_2.to_csv("de-en/deck-2.csv", sep="|")


partition_deck_data()

# %%


def render_ankiweb_descriptions():
    def mk_template_path(x: str):
        return f"de-en/anki-deck/audio-files/ankiweb-description{x}.html"

    def read_template():
        with open(mk_template_path(""), "r", encoding="UTF-8") as t:
            return t.read()

    template = read_template()

    def write_template(template_path_suff: str, replacement: str):
        with open(mk_template_path(template_path_suff), "w", encoding="UTF-8") as t:
            t.write(template.replace("{{n_words}}", replacement))

    write_template("-1", "the 1st to the 3000th")
    write_template("-2", "the 3001st to the last")


render_ankiweb_descriptions()
