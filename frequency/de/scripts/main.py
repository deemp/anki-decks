# %%

import os
import pandas as pd
from IPython.display import display

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

FREQUENCY_CUTOFF_1 = 1700
FREQUENCY_CUTOFF_2 = 3400

# %%


def check(deck: pd.DataFrame):
    duplicates = deck[deck.index.duplicated()]

    if not duplicates.empty:
        display(duplicates)
        raise (Exception("Duplicates!"))

    FRACTIONAL_PART_MAX = 0.01

    indices_fractional = deck[
        (deck.index % 1 != 0) & (deck.index % 1 >= FRACTIONAL_PART_MAX)
    ]

    if not indices_fractional.empty:
        display(indices_fractional)
        raise (Exception(f"Fractional parts not less than {FRACTIONAL_PART_MAX}!"))


def partition_deck_data():
    deck_full = pd.read_csv("de-en/deck.csv", sep="|", index_col=0)

    check(deck=deck_full)

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

    ids = [1848185140, 186639246, 1082248180]

    n_words = [
        f"the 1st to the {FREQUENCY_CUTOFF_1}th",
        f"the {FREQUENCY_CUTOFF_1 + 1}st to the {FREQUENCY_CUTOFF_2}th",
        f"the {FREQUENCY_CUTOFF_2 + 1}st to the {max_frequency}th",
    ]

    for i, n_words_cur in enumerate(n_words):
        another_part_ankiweb_ids = [k for j, k in enumerate(ids) if j != i]

        this_part_number = i + 1
        another_part_numbers = [k for k in [1, 2, 3] if k != this_part_number]

        replacement = (
            {"n_words": n_words_cur, "deck_index": str(this_part_number)}
            | {
                f"another_part_ankiweb_id_{i + 1}": str(another_part_ankiweb_id)
                for i, another_part_ankiweb_id in enumerate(another_part_ankiweb_ids)
            }
            | {
                f"another_part_number_{i + 1}": str(another_part_number)
                for i, another_part_number in enumerate(another_part_numbers)
            }
        )

        write_template(
            template_path_suff=f"-{this_part_number}",
            replacement=replacement,
        )


render_ankiweb_descriptions()
