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

    def write_template(
        template: str, template_path_suff: str, replacement: dict[str, str]
    ):
        for k, v in replacement.items():
            template = template.replace(f"{{{{{k}}}}}", v)
        with open(mk_template_path(template_path_suff), "w", encoding="UTF-8") as t:
            t.write(template)

    write_template(
        str(template),
        "-1",
        {"n_words": "the 1st to the 3000th", "deck_id": "1946034909"},
    )
    write_template(
        str(template),
        "-2",
        {"n_words": "the 3001st to the 5000th", "deck_id": "763225563"},
    )


render_ankiweb_descriptions()
