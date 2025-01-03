# %%

import os
import pandas as pd

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

FREQUENCY_CUTOFF_1 = 1700
FREQUENCY_CUTOFF_2 = 3400

# %%


def partition_deck_data():
    os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

    deck_full = pd.read_csv("de-en/deck.csv", sep="|", index_col=0)
    deck_full.index = deck_full.index.map(lambda x: x if x % 1 != 0 else f"{int(x)}")

    frequency_rank = deck_full["frequency_rank"]

    deck_part_1 = deck_full.loc[frequency_rank <= FREQUENCY_CUTOFF_1]
    deck_part_2 = deck_full.loc[
        (frequency_rank > FREQUENCY_CUTOFF_1) & (frequency_rank <= FREQUENCY_CUTOFF_2)
    ]
    deck_part_3 = deck_full.loc[frequency_rank > FREQUENCY_CUTOFF_2]

    for i, deck in enumerate([deck_part_1, deck_part_2, deck_part_3]):
        deck.to_csv(f"de-en/deck-{i + 1}.csv", sep="|")


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
        template_path_suff: str,
        replacement: dict[str, str],
        template: str = str(template),
    ):
        for k, v in replacement.items():
            template = template.replace(f"{{{{{k}}}}}", v)
        with open(mk_template_path(template_path_suff), "w", encoding="UTF-8") as t:
            t.write(template)

    deck_full = pd.read_csv("de-en/deck.csv", sep="|", index_col=0)
    max_frequency = max(deck_full["frequency_rank"])

    ids = ["1848185140", "186639246", "1082248180"]

    n_words = [
        f"the 1st to the {FREQUENCY_CUTOFF_1}th",
        f"the {FREQUENCY_CUTOFF_1 + 1}st to the {FREQUENCY_CUTOFF_2}th",
        f"the {FREQUENCY_CUTOFF_1 + 1}st to the {max_frequency}th",
    ]

    for i, n_words_cur in enumerate(n_words):
        ids_cur = [k for j, k in enumerate(ids) if j != i]

        replacement = {"n_words": n_words_cur} | {
            f"another_part_id_{j + 1}": id_cur for j, id_cur in enumerate(ids_cur)
        }

        write_template(
            template_path_suff=f"-{i + 1}",
            replacement=replacement,
        )


render_ankiweb_descriptions()
