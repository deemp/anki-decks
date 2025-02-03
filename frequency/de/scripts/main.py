# %%

import os
import re
from functools import cmp_to_key
from pathlib import Path
import pandas as pd
from IPython.display import display
import spacy

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')
nlp = spacy.load("de_dep_news_trf")

# %%

FREQUENCY_CUTOFF_1 = 1700
FREQUENCY_CUTOFF_2 = 3400


class PATH:
    PWD = Path(".")
    DE_EN = PWD / "de-en"
    DATA = PWD / "de-en" / "data"
    DECK = DE_EN / "deck.csv"
    EXTRA = DATA / "extra.csv"
    DECK_RAW = DATA / "deck-raw.csv"
    WORD_COUNT = DATA / "word-count.csv"
    WORDS_BAD_BASEFORM = DATA / "words-bad-baseform.csv"
    WORDS_TOO_FREQUENT = DATA / "words-too-frequent.csv"


class DECK_PART_START_INDEX:
    P1 = 0
    # frequency rank 1700
    P2 = 1783
    # frequency rank 3400
    P3 = 3501


class INDEX_SUFFIX:
    ALTERNATIVE_MEANING = 0.001
    NEW_WORD = 0.0001


class SENTENCE_LENGTH:
    # min and max for each deck part
    MIN1 = 40
    MAX1 = 60
    MIN2 = 60
    MAX2 = 80
    MIN3 = 80
    MAX3 = 100


ARTICLES_FULL = ["die", "der", "das"]

LEMMATIZED_SEP = ";"

MAX_OCCURENCES = 4


def normalize_index_single(index: float):
    return index if index % 1 != 0 else f"{int(index)}"


def normalize_index(df: pd.DataFrame):
    df.index = df.index.map(normalize_index_single)


def remove_separators_in_file(path: Path):
    with open(path, "r", encoding="UTF-8") as d:
        deck_lines = d.readlines()

    for i, line in enumerate(deck_lines):
        for j in range(1, 10):
            if line[-j] not in ["|", "\n"]:
                deck_lines[i] = line[: -j + 1] + "\n"
                break

    with open(path, "w", encoding="UTF-8") as d:
        d.writelines(deck_lines)

def write_deck(deck: pd.DataFrame()):
    normalize_index(deck)
    deck.to_csv(PATH.DECK, sep="|")
    remove_separators_in_file(path=PATH.DECK)


def write_gen_data(gen_data: pd.DataFrame()):
    gen_data.to_csv(PATH.DECK_RAW, sep="|")
    remove_separators_in_file(path=PATH.DECK_RAW)


def normalize_verb(x: pd.DataFrame):
    word_en = x["word_en"]
    return (
        f"to {word_en}"
        if x["part_of_speech"] == "verb"
        and not pd.isna(word_en)
        and not word_en.startswith("to ")
        else word_en
    )


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


def add_rows_for_alternative_translations():
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


def copy_deck_to_deck_raw():
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)

    columns = [
        "word_de",
        "part_of_speech",
        "word_en",
    ]
    if not Path(PATH.DECK_RAW).is_file():
        pd.DataFrame(columns=columns).to_csv(PATH.DECK_RAW, sep="|")

    deck_raw = pd.read_csv(PATH.DECK_RAW, sep="|", index_col=0)

    deck_missing_rows = deck.loc[~deck.index.isin(deck_raw.index), columns]

    deck_raw.loc[:, columns] = deck.loc[deck_raw.index, columns].values

    deck_raw = pd.concat([deck_raw, deck_missing_rows])

    write_gen_data(gen_data=deck_raw)


def make_baseform(word: str) -> str:
    # article
    if word in ARTICLES_FULL:
        return word
    for i, j in enumerate(word):
        if j in [".", "(", ","]:
            word = word[:i].strip()
            break
    # noun or name
    if word[0].isupper():
        return word
    # noun with an article
    if word[:3] in ARTICLES_FULL and word[3] == " ":
        for i, j in enumerate(word):
            if j.isupper():
                word = word[i:]
                break
        return word
    return word


def is_noun(word: str):
    return make_baseform(word)[0].isupper()


def tokenize_sentence(sentence: str):
    doc = nlp(sentence)
    tokens = [tok.lemma_ for tok in doc]

    # separable verbs
    for token in doc:
        if token.dep_ == "svp" and token.head.pos_ == "VERB":
            verb_stem = token.head.lemma_
            prefix = token.text
            tokens[token.head.i] = prefix + verb_stem

    tokens = [tok for tok in tokens if tok not in ["-", "--", " ", "  "]]

    return LEMMATIZED_SEP.join(tokens)


def update_gen_data_lemmatized_sentences(gen_data: pd.DataFrame()):
    not_lemmatized_cond = gen_data["sentence_lemmatized_de"].isna()

    gen_data.loc[not_lemmatized_cond, "sentence_lemmatized_de"] = gen_data.loc[
        not_lemmatized_cond, "sentence_de"
    ].map(tokenize_sentence, na_action="ignore")

    write_gen_data(gen_data=gen_data)

    return gen_data


def update_word_counts(gen_data=pd.DataFrame()):
    words = pd.DataFrame(
        gen_data["sentence_lemmatized_de"]
        .map(lambda x: x.split(";"), na_action="ignore")
        .explode("sentence_lemmatized_de")
    ).rename(columns={"sentence_lemmatized_de": "word_de"})

    word_counts = pd.DataFrame(words.value_counts())
    word_counts.to_csv(PATH.WORD_COUNT, sep="|")

    return word_counts


def check_is_correct_sentence(
    row: pd.Series(),
    word_stats: pd.DataFrame(),
    words_bad_wordforms: pd.DataFrame(),
    words_too_frequent: pd.DataFrame(),
) -> bool:
    word_de = row["word_de"]
    word_de = (
        make_baseform(word_de)
        if row.name not in words_bad_wordforms.index
        else words_bad_wordforms.loc[
            words_bad_wordforms["word"] == word_de, "baseform"
        ].values[0]
    )

    sentence_lemmatized_de = row["sentence_lemmatized_de"].split(LEMMATIZED_SEP)

    word_is_in_the_sentence = (
        # True
        word_de
        in sentence_lemmatized_de
    )

    words = pd.DataFrame(data=sentence_lemmatized_de, columns=["word_de"])

    word_counts = words.join(word_stats, on="word_de", rsuffix="r")

    # run this check on all sentences
    # because the number of sentences will grow
    # and the word counts will grow too
    sentence_contains_rare_words = any(
        (word_counts["word_de"] != word_de) & (word_counts["count"] <= MAX_OCCURENCES)
    )

    # sentence_contains_no_too_frequent_words = [x for x in sentence_lemmatized_de if x in words_too_frequent.index] == []

    result = sentence_contains_rare_words & word_is_in_the_sentence
    # & sentence_contains_no_too_frequent_words

    # I subtract to correctly analyze other rows
    if not result:
        for word in words["word_de"]:
            word_stats.loc[word, "count"] -= 1

    return result


def partition_gen_data_by_having_sentence_de(gen_data: pd.DataFrame()):
    has_data_cond = gen_data["sentence_de"].notna()
    rows_with_data = gen_data[has_data_cond].sort_index()
    rows_without_data = gen_data[~has_data_cond].sort_index()

    gen_data = pd.concat([rows_with_data, rows_without_data])
    write_gen_data(gen_data=gen_data)

    return rows_with_data, rows_without_data


def get_sentence_length_bounds(idx: float):
    if idx < DECK_PART_START_INDEX.P2:
        return SENTENCE_LENGTH.MIN1, SENTENCE_LENGTH.MAX1
    if idx < DECK_PART_START_INDEX.P3:
        return SENTENCE_LENGTH.MIN2, SENTENCE_LENGTH.MAX2
    return SENTENCE_LENGTH.MIN3, SENTENCE_LENGTH.MAX3


def check_sentence_length(x: pd.Series()):
    idx = x.name
    mini, maxi = get_sentence_length_bounds(idx=idx)
    return pd.notna(x["sentence_de"]) and mini <= len(x["sentence_de"]) <= maxi


def filter_gen_data_by_sentence_length(gen_data: pd.DataFrame()):
    rows_with_data, rows_without_data = partition_gen_data_by_having_sentence_de(
        gen_data=gen_data
    )

    is_good_sentence_length_cond = rows_with_data.apply(check_sentence_length, axis=1)
    rows_with_good_sentence_length = rows_with_data[is_good_sentence_length_cond]
    rows_with_bad_sentence_length = pd.DataFrame(
        rows_with_data.loc[
            ~is_good_sentence_length_cond, ["word_de", "part_of_speech", "word_en"]
        ]
    )

    rows_without_data = pd.concat(
        [rows_with_bad_sentence_length, rows_without_data]
    ).sort_index()

    gen_data = pd.concat([rows_with_good_sentence_length, rows_without_data])

    write_gen_data(gen_data=gen_data)

    return gen_data


def partition_deck_raw(gen_data: pd.DataFrame()):
    gen_data = filter_gen_data_by_sentence_length(gen_data=gen_data)
    gen_data = update_gen_data_lemmatized_sentences(gen_data=gen_data)
    word_stats = update_word_counts(gen_data=gen_data)
    rows_with_data, rows_without_data = partition_gen_data_by_having_sentence_de(
        gen_data=gen_data
    )

    words_bad_wordforms = pd.read_csv(PATH.WORDS_BAD_BASEFORM, sep="|", index_col=0)
    words_too_frequent = pd.read_csv(PATH.WORDS_TOO_FREQUENT, sep="|", index_col=0)

    # prefer removing rows with a larger index
    # to change rows with smaller index less frequently
    # to preserve progress
    #
    # rows with smaller indices may get removed
    # if we leave rows with larger indices that introduced some words
    # that increased word counts
    # and hence disallowed rows with smaller indices to stay
    is_correct_sentence_cond = (
        rows_with_data.iloc[::-1]
        .apply(
            lambda row: check_is_correct_sentence(
                row=row,
                word_stats=word_stats,
                words_bad_wordforms=words_bad_wordforms,
                words_too_frequent=words_too_frequent,
            ),
            axis=1,
        )
        .iloc[::-1]
    )

    has_correct_sentence = rows_with_data[is_correct_sentence_cond]
    has_incorrect_sentence = rows_with_data[~is_correct_sentence_cond]
    has_incorrect_sentence = has_incorrect_sentence[
        ["word_de", "part_of_speech", "word_en"]
    ]

    rows_without_data = pd.concat(
        [has_incorrect_sentence, rows_without_data]
    ).sort_index()

    gen_data = pd.concat([has_correct_sentence, rows_without_data])
    gen_data = gen_data[~gen_data.index.duplicated()]

    write_gen_data(gen_data=gen_data)

    return gen_data


def update_gen_data():
    gen_data = pd.read_csv(PATH.DECK_RAW, sep="|", index_col=0)
    partition_deck_raw(gen_data=gen_data)


def copy_generated_to_deck():
    gen_data = pd.read_csv(PATH.DECK_RAW, sep="|", index_col=0)
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)
    has_sentence_cond = gen_data["sentence_de"].notna()
    columns = ["sentence_de", "sentence_en", "sentence_lemmatized_de"]

    display(
        deck.index[~deck.index.isin(gen_data.index)],
        gen_data.index[~gen_data.index.isin(deck.index)],
    )

    deck.loc[has_sentence_cond, columns] = gen_data.loc[
        has_sentence_cond, columns
    ].values

    write_deck(deck=deck)


# %%
copy_deck_to_deck_raw()

# %%
update_gen_data()

# %%

copy_generated_to_deck()
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
