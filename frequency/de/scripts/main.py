# %%

import os
import re
import pandas as pd
from IPython.display import display

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')

FREQUENCY_CUTOFF_1 = 1700
FREQUENCY_CUTOFF_2 = 3400


def normalize_index(df: pd.DataFrame):
    def go(index_val: float):
        return index_val if index_val % 1 != 0 else f"{int(index_val)}"

    df.index = df.index.map(go)


# %%


def check_deck_data(deck: pd.DataFrame):
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

    check_deck_data(deck=deck_full)

    normalize_index(deck_full)

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

        if patterns := re.findall(pattern=r"\{\{[^}]*\}\}", string=template):
            raise Exception(f"These template patterns weren't replaced: {patterns}")

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

# %%


def find_not_all_meanings():
    path = "de-en/deck.csv"

    deck_full = pd.read_csv(path, sep="|", index_col=0)

    for i in deck_full.index:
        real_index = deck_full.index.get_loc(i)
        if int(i) == i:
            row = deck_full[deck_full.index == i]
            word_translations_en = row["word_translations_en"].values[0]
            if type(word_translations_en) == str and "," in word_translations_en:
                word_de = row["word_de"].values[0]
                translations = list(
                    map(lambda x: x.strip(), word_translations_en.split(","))
                )
                block = deck_full[(deck_full.index >= i) & (deck_full.index < i + 1)]

                last_index = block.index.max()

                for translation in translations:
                    row_translation = block[block["word_en"] == translation]
                    if row_translation.empty:
                        last_index += 0.001

                        if last_index % 1 >= 0.01:
                            raise Exception(f"Bad {last_index=}")

                        print(
                            f"{last_index:.3f}|{row["part_of_speech"].values[0]}|{word_de}|{translation}"
                        )


find_not_all_meanings()

# %%


def reorder_not_all_meanings():
    mk_path = lambda x: f"de-en/data/translations-without-example{x}.csv"
    path1 = mk_path("")
    path2 = mk_path("-reordered")

    deck = pd.read_csv(path1, sep="|", index_col=0)

    cond = deck["sentence_de"].apply(lambda x: 30 <= len(str(x)) <= 45)

    deck_1 = pd.DataFrame(deck[cond]).sort_index()

    deck_2 = pd.DataFrame(deck[~cond]).sort_index()
    deck_2 = deck_2.drop(columns=["sentence_de", "sentence_en"])

    deck_new = pd.concat([deck_1, deck_2])

    deck_new.to_csv(path2, sep="|")

    with open(path2, "r", encoding="UTF-8") as p2:
        lines = p2.readlines()
    for i, line in enumerate(lines):
        if line[-3:] == "||\n":
            lines[i] = line[:-3] + "\n"
    with open(path2, "w", encoding="UTF-8") as p2:
        p2.writelines(lines)


reorder_not_all_meanings()


# %%


def cmp(x, y):
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0


def leq(x, y):

    if int(x) != int(y):
        return cmp(x, y)

    x = round(x % 1, 6)
    y = round(y % 1, 6)

    if x == 0 or y == 0:
        return cmp(x, y)

    while x < 1 and y < 1:
        x *= 10
        y *= 10

    if x >= 1 and y >= 1:
        return cmp(x, y)
    else:
        return cmp(y, x)


print(
    [
        leq(2.003, 1.002),
        leq(2.001, 2.002),
        leq(2.001, 2.001),
        leq(2.001, 2.0002),
        leq(2.00132, 2.00130),
        leq(2.00132, 2.00132),
        leq(2.00132, 2.00135),
    ]
)

# %%

from functools import cmp_to_key


def copy_good_examples():
    path_good_sentences = "de-en/data/translations-without-example-good-sentences.csv"
    deck_good_sentences = pd.read_csv(path_good_sentences, sep="|", index_col=0)

    path = "de-en/deck.csv"
    deck = pd.read_csv(path, sep="|", index_col=0)

    deck_good_sentences = deck_good_sentences.drop(columns=["part_of_speech"])

    skipped_indices = [
        174,
        251,
        667,
        702,
        720,
        727,
    ]

    mismatches = pd.Index(
        [
            idx
            for idx in deck_good_sentences.index
            if int(idx) not in skipped_indices
            and deck_good_sentences.loc[idx, "word_de"] != deck.loc[int(idx), "word_de"]
        ]
    )

    if not mismatches.empty:
        deck_mismatches = pd.DataFrame(deck_good_sentences.loc[mismatches])

        word_de_actual = deck.loc[deck_mismatches.index.map(int), "word_de"].values

        display(word_de_actual)

        deck_mismatches.insert(
            0,
            column="actual_word",
            value=word_de_actual,
        )

        display(
            deck_mismatches.loc[
                :,
                [
                    "word_de",
                    "actual_word",
                ],
            ]
        )

        raise Exception(f"Mismatched rows!")

    deck_matching_good_sentences = pd.DataFrame(
        deck.loc[deck_good_sentences.index.map(int)]
    )

    deck_matching_good_sentences.index = deck_good_sentences.index

    deck_matching_good_sentences.loc[
        :, ["word_de", "sentence_de", "word_en", "sentence_en"]
    ] = deck_good_sentences.loc[:, ["word_de", "sentence_de", "word_en", "sentence_en"]]

    deck_updated = pd.concat([deck, deck_matching_good_sentences])

    sorted_index = pd.Index(sorted(deck_updated.index, key=cmp_to_key(leq)))

    deck_updated_sorted = deck_updated.reindex(sorted_index)

    normalize_index(deck_updated_sorted)

    deck_updated_sorted.to_csv(path, sep="|")


copy_good_examples()

# %%


def check_meanings_have_the_same_word_de():
    skipped_indices = [
        174,
        251,
        667,
        702,
        720,
        727,
        986,
        1069,
        1220,
        1295,
        1300,
        1314,
        1363,
    ]

    path = "de-en/deck.csv"
    deck = pd.read_csv(path, sep="|", index_col=0)

    mismatches = pd.Index(
        [
            idx
            for idx in deck.index
            if int(idx) not in skipped_indices
            and deck.loc[idx, "word_de"] != deck.loc[int(idx), "word_de"]
        ]
    )

    deck_root = deck.loc[mismatches.map(int)]
    deck_mismatched = deck.loc[mismatches]

    if not deck_mismatched.empty:
        deck_res = pd.concat([deck_root, deck_mismatched])

        display(deck_res.sort_index().drop_duplicates())

        raise Exception("Mismatches in word_de found!")


check_meanings_have_the_same_word_de()
