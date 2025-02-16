# %%

import os
import re
from functools import cmp_to_key
from pathlib import Path
import importlib
import pandas as pd
from IPython.display import display
import spacy

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')
nlp = spacy.load("de_dep_news_trf")

# %%

import custom.de.script.lib as lib

importlib.reload(lib)

from custom.de.script.lib import (
    remove_separators_in_file,
    leq,
    update_deck_raw,
    write_deck_raw,
    APIRequestsArgs,
    MODEL,
    generate_deck_data_iteratively,
)


FREQUENCY_CUTOFF_1 = 1700
FREQUENCY_CUTOFF_2 = 3400


class PATH:
    PWD = Path(".")
    DE_EN = PWD / "de-en"
    DATA = PWD / "de-en" / "data"
    DECK = DE_EN / "deck.csv"
    EXTRA = DATA / "extra.csv"
    DECK_RAW = DATA / "deck-raw-gpt-4o-2024-11-20-length-60-70.csv"
    WORD_COUNT = DATA / "word-count.csv"
    WORDS_BAD_BASEFORM = DATA / "words-bad-baseform.csv"
    WORDS_TOO_FREQUENT = DATA / "words-too-frequent.csv"
    PARALLEL_REQUESTS = DATA / "parallel-requests.jsonl"
    PARALLEL_RESPONSES = DATA / "parallel-responses.jsonl"
    PARALLEL_RESPONSES_CONCATENATED = DATA / "parallel-responses-concatenated.csv"


class DECK_PART_START_INDEX:
    P1 = 0
    # frequency rank 1700
    P2 = 1783
    # frequency rank 3400
    P3 = 3501


class INDEX_SUFFIX:
    ALTERNATIVE_MEANING = 0.001
    NEW_WORD = 0.0001


ARTICLES_FULL = ["die", "der", "das"]

LEMMATIZED_SEP = ";"

MIN_WORDS = 2

MAX_OCCURENCES = 3


class PART:
    ITERATIONS = 2
    SIZE = 30
    COUNT_PER_ITERATION = 2


class SENTENCE_LENGTH:
    MIN = 60
    MAX = 70


def normalize_index_single(index: float):
    return index if index % 1 != 0 else f"{int(index)}"


def normalize_index(df: pd.DataFrame):
    df.index = df.index.map(normalize_index_single)


def write_deck(deck: pd.DataFrame()):
    normalize_index(deck)
    deck.to_csv(PATH.DECK, sep="|")
    remove_separators_in_file(path=PATH.DECK)


def normalize_verb(x: pd.DataFrame):
    word_en = x["word_en"]
    return (
        f"to {word_en}"
        if x["part_of_speech"] == "verb"
        and not pd.isna(word_en)
        and not word_en.startswith("to ")
        else word_en
    )


def add_deck_rows_for_alternative_translations():
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)

    custom_row_cond = deck.index.map(
        lambda x: INDEX_SUFFIX.NEW_WORD
        <= round(x % 1, ndigits=4)
        < INDEX_SUFFIX.ALTERNATIVE_MEANING
    )
    deck_custom_rows = deck[custom_row_cond]
    deck_not_custom_rows = deck[~custom_row_cond]

    multiple_translations_cond = deck_not_custom_rows["word_translations_en"].map(
        lambda x: pd.notna(x) and "," in x,
    )
    deck_single_translation_rows = deck_not_custom_rows[~multiple_translations_cond]
    deck_multiple_translations_rows = pd.DataFrame(
        deck_not_custom_rows[multiple_translations_cond]
    )
    deck_multiple_translations_rows["word_en"] = deck_multiple_translations_rows.apply(
        normalize_verb, axis=1
    )

    main_translation_cond = deck_multiple_translations_rows.index.map(
        lambda x: round(x % 1, ndigits=4) == 0
    )
    deck_main_translation_rows = deck_multiple_translations_rows[main_translation_cond]

    deck_all_translations_rows = pd.DataFrame(deck_main_translation_rows)
    deck_all_translations_rows["word_en"] = deck_all_translations_rows[
        "word_translations_en"
    ].map(lambda x: [y.strip() for y in x.split(",")], na_action="ignore")
    deck_all_translations_rows = deck_all_translations_rows.explode("word_en")
    deck_all_translations_rows["word_en"] = deck_all_translations_rows.apply(
        normalize_verb, axis=1
    )

    index_new = []
    counts = {}
    for idx in deck_all_translations_rows.index:
        if idx not in counts:
            counts[idx] = 0
            index_new.append(idx)
        else:
            counts[idx] += 1
            index_new.append(idx + INDEX_SUFFIX.ALTERNATIVE_MEANING * counts[idx])

    deck_all_translations_rows.index = index_new

    extra_rows = deck_all_translations_rows[
        deck_all_translations_rows.index.map(lambda x: False)
    ]

    for idx in deck_main_translation_rows.index:
        block_all_translations_cond = (idx <= deck_all_translations_rows.index) & (
            deck_all_translations_rows.index < idx + 1
        )

        block_all_translations = deck_all_translations_rows.loc[
            block_all_translations_cond
        ]

        block_existing_translations = deck_multiple_translations_rows.loc[
            (idx <= deck_multiple_translations_rows.index)
            & (deck_multiple_translations_rows.index < idx + 1)
        ]

        block_existing_translations = block_existing_translations[
            ~block_existing_translations.index.duplicated()
        ]

        block_existing_translations_unique = block_existing_translations[
            ["word_en", "sentence_de", "sentence_en"]
        ].set_index("word_en")

        block_existing_translations_unique = block_existing_translations_unique[
            ~block_existing_translations_unique.index.duplicated()
        ]

        block_all_translations = block_all_translations.join(
            block_existing_translations_unique,
            on="word_en",
            rsuffix="_r",
        )

        block_all_translations.loc[:, ["sentence_de", "sentence_en"]] = (
            block_all_translations.loc[:, ["sentence_de_r", "sentence_en_r"]].values
        )

        block_all_translations.drop(
            columns=["sentence_de_r", "sentence_en_r"], inplace=True
        )

        deck_all_translations_rows.loc[block_all_translations_cond] = (
            block_all_translations
        )

        block_extra_translations = block_existing_translations[
            ~block_existing_translations["sentence_de"].isin(
                block_all_translations["sentence_de"]
            )
        ]

        extra_rows = pd.concat([extra_rows, block_extra_translations])

    deck_updated = pd.concat([deck_single_translation_rows, deck_all_translations_rows])

    sorted_index = pd.Index(sorted(deck_updated.index, key=cmp_to_key(leq)))
    deck_updated = deck_updated.reindex(sorted_index)

    write_deck(deck=deck_updated)

    extra = pd.concat([deck_custom_rows, extra_rows])
    extra.to_csv(PATH.EXTRA, sep="|")


def copy_deck_to_deck_raw(deck_path: type[Path], deck_raw_path: type[Path]):
    deck = pd.read_csv(deck_path, sep="|", index_col=0)

    columns = [
        "word_de",
        "part_of_speech",
        "word_en",
    ]
    if not Path(deck_raw_path).is_file():
        pd.DataFrame(columns=columns).to_csv(deck_raw_path, sep="|")

    deck_raw = pd.read_csv(deck_raw_path, sep="|", index_col=0)

    deck_missing_rows = deck.loc[~deck.index.isin(deck_raw.index), columns]

    deck_raw.loc[:, columns] = deck.loc[deck_raw.index, columns].values

    deck_raw = pd.concat([deck_raw, deck_missing_rows])

    write_deck_raw(deck_raw=deck_raw, deck_raw_path=deck_raw_path)

    return deck_raw


def copy_deck_raw_to_deck():
    deck_raw = pd.read_csv(PATH.DECK_RAW, sep="|", index_col=0)
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)
    has_sentence_cond = deck_raw["sentence_de"].notna()
    columns = ["sentence_de", "sentence_en", "sentence_lemmatized_de"]

    display(
        deck.index[~deck.index.isin(deck_raw.index)],
        deck_raw.index[~deck_raw.index.isin(deck.index)],
    )

    deck.loc[has_sentence_cond, columns] = deck_raw.loc[
        has_sentence_cond, columns
    ].values

    write_deck(deck=deck)


async def generate_deck_data():
    print("Starting update")

    copy_deck_to_deck_raw(deck_path=PATH.DECK, deck_raw_path=PATH.DECK_RAW)

    api_requests_args = APIRequestsArgs(
        requests_filepath=f"{PATH.DATA}/parallel-requests.jsonl",
        save_filepath=f"{PATH.DATA}/parallel-responses.jsonl",
        max_attempts=3,
        request_url="https://api.openai.com/v1/chat/completions",
    )

    await update_deck_raw(
        word_counts_path=PATH.WORD_COUNT,
        deck_raw_path=PATH.DECK_RAW,
        max_occurences=MAX_OCCURENCES,
        words_bad_baseform_path=PATH.WORDS_BAD_BASEFORM,
        part_count_per_iteration=PART.COUNT_PER_ITERATION,
        part_size=PART.SIZE,
        parallel_requests_path=PATH.PARALLEL_REQUESTS,
        parallel_responses_path=PATH.PARALLEL_RESPONSES,
        parallel_responses_concatenated_path=PATH.PARALLEL_RESPONSES_CONCATENATED,
        api_requests_args=api_requests_args,
        model=MODEL.CHATGPT_4O_MINI,
        nlp=nlp,
        sentence_length_min=SENTENCE_LENGTH.MIN,
        sentence_length_max=SENTENCE_LENGTH.MAX,
        has_part_of_speech=True,
        has_word_en=True,
    )

    copy_deck_raw_to_deck()

    print("Update completed!")


# %%

add_deck_rows_for_alternative_translations()

# %%

await generate_deck_data_iteratively(
    generate_deck_data=generate_deck_data, iterations=PART.ITERATIONS
)

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

    skipped_indices = [87, 174, 251, 667, 702, 720, 727, 1727]

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

    if not sorted_index[sorted_index.duplicated()].empty:
        display(sorted_index[sorted_index.duplicated()])
        raise Exception("Duplicates!")

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

# %%


def check_verbs_have_to_prefix():
    path = "de-en/deck.csv"
    deck = pd.read_csv(path, sep="|", index_col=0)

    bad_verbs = deck[
        (deck["part_of_speech"] == "verb")
        & deck["word_en"].map(lambda x: not str(x).startswith("to"))
    ]

    skipped_indices = [34, 59, 78]

    display(
        bad_verbs.loc[
            [idx for idx in bad_verbs.index if int(idx) not in skipped_indices]
        ]
    )


check_verbs_have_to_prefix()
