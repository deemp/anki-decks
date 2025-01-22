# %%

import os
from pathlib import Path
import pandas as pd
from IPython.display import display
import yaml

os.chdir(f'{os.environ["ROOT_DIR"]}/custom/de')

ARTICLES_SHORT = ["f", "m", "n"]
ARTICLES_FULL = ["die", "der", "das"]
ARTICLES_SEP = "/"
DEWIKI_ARTICLES_SEP = ";"
LYRICS_LEMMATIZED_SEP = ";"


class PATH:
    DATA = Path("data")
    PWD = Path(".")
    DECK = PWD / "deck.csv"
    SOURCES = DATA / "sources"
    SOURCES_LEMMATIZED = SOURCES / "lemmatized.csv"
    SOURCES_TEXTS = SOURCES / "texts.yaml"
    SOURCES_WORDS_LEMMAS = SOURCES / "words-lemmas.csv"
    SOURCES_WORDS_NOT_LEMMAS = SOURCES / "words-not-lemmas.csv"
    SOURCES_WORDS = SOURCES / "words.csv"


class INDEX_SUFFIX:
    ALTERNATIVE_MEANING = 0.001
    NEW_WORD = 0.0001


class SENTENCE_LENGTH:
    MIN = 30
    MAX = 50


def mk_word(word: str):
    if word in ARTICLES_FULL:
        return word
    if any(x.isupper() for x in word):
        k = 0
        for i, j in enumerate(word):
            k = i
            if j.isupper():
                break
        word = word[k:]
    return word


def is_noun(word: str):
    return mk_word(word)[0].isupper()


ARTICLES_DICT = dict(zip(ARTICLES_SHORT, ARTICLES_FULL))

article_mapping = pd.DataFrame(
    data=ARTICLES_FULL,
    columns=["full"],
    index=pd.Index(ARTICLES_SHORT, name="articles"),
)

dewiki_noun_articles = pd.read_csv("data/dewiki-noun-articles.csv", sep="|")


def get_lemmas_articles(nouns: pd.DataFrame):
    lemma = "lemma"
    lemmas_initial = pd.DataFrame(nouns[lemma].map(mk_word))
    lemmas_with_articles = lemmas_initial.join(
        other=dewiki_noun_articles.set_index(lemma), on=lemma
    )
    return lemmas_with_articles


lemmata = pd.read_csv("data/dwds_lemmata_2025-01-15.csv")


def mk_is_lemma_cond(df: pd.DataFrame):
    return df.isin(lemmata["lemma"]) | df.isin(dewiki_noun_articles["lemma"])


def remove_separators(path: Path, n: int):
    with open(path, "r", encoding="UTF-8") as d:
        deck_text = d.read()

    with open(path, "w", encoding="UTF-8") as d:
        d.write(deck_text.replace("|" * n, ""))


# %%

import spacy

nlp = spacy.load("de_dep_news_trf")

# %%


def find_tokens(sentence: str):
    doc = nlp(sentence)
    tokens = [tok.lemma_ for tok in doc]

    # separable verbs
    for token in doc:
        if token.dep_ == "svp" and token.head.pos_ == "VERB":
            verb_stem = token.head.lemma_
            prefix = token.text
            tokens[token.head.i] = prefix + verb_stem

    tokens = [tok for tok in tokens if tok not in ["-", "--", " ", "  "]]

    return tokens


# %%


def update_lemmatized_sources():
    sources_lemmatized = pd.read_csv(PATH.SOURCES_LEMMATIZED, sep="|", index_col=0)

    with open(PATH.SOURCES_TEXTS, mode="r", encoding="UTF-8") as f:
        sources = yaml.safe_load(f)

    sources_new = pd.DataFrame(
        sources,
        columns=["author", "title", "text"],
    ).astype(pd.StringDtype())

    sources_new = sources_new[~sources_new.index.isin(sources_lemmatized.index)]

    sources_new["text"] = sources_new["text"].map(
        lambda x: "".join(
            [
                y
                for y in x.replace("\n", " ")
                if y.isalpha() or y.isnumeric() or y in [" ", "-"]
            ]
        )
    )

    sources_new["text"] = sources_new["text"].map(
        lambda x: (LYRICS_LEMMATIZED_SEP.join(find_tokens(x)))
    )

    sources_lemmatized = pd.concat([sources_lemmatized, sources_new])

    sources_lemmatized.to_csv(PATH.SOURCES_LEMMATIZED, sep="|")


update_lemmatized_sources()
# %%


def update_word_lists():
    # save all words

    lyrics_lemmatized = pd.read_csv(PATH.SOURCES_LEMMATIZED, sep="|", index_col=0)

    texts = lyrics_lemmatized[["text"]]
    texts.loc[:, "text"] = texts.loc[:, "text"].map(
        lambda x: str(x).split(LYRICS_LEMMATIZED_SEP)
    )
    texts = texts.rename(columns={"text": "word"})

    words = texts.explode("word")
    words.reset_index(names="song_id", inplace=True)
    words = words[["song_id", "word"]]
    words.to_csv(PATH.SOURCES_WORDS, sep="|")

    words = words.loc[~words.duplicated("word"), ["word"]]

    is_lemma_cond = mk_is_lemma_cond(words["word"])

    # update non-lemmas

    words_not_lemmas_new = pd.DataFrame(words[~is_lemma_cond])
    words_not_lemmas_existing = pd.read_csv(
        PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|", index_col=0
    )
    words_not_lemmas = words_not_lemmas_new.join(
        other=words_not_lemmas_existing.set_index("word"), on="word"
    )
    words_not_lemmas.sort_index(inplace=True)
    words_not_lemmas = words_not_lemmas[~words_not_lemmas.duplicated("word")]

    has_lemma_cond = words_not_lemmas["lemma"].notna()

    words_not_lemmas = pd.concat(
        [
            words_not_lemmas[has_lemma_cond].sort_index(),
            words_not_lemmas[~has_lemma_cond].sort_index(),
        ]
    )

    words_not_lemmas.to_csv(PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|")

    # update lemmas

    lemmas_new = pd.DataFrame(words[is_lemma_cond])
    lemmas_new = lemmas_new.rename(columns={"word": "lemma"})

    lemmas_existing = pd.read_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|", index_col=0)

    lemmas = lemmas_new.join(
        lemmas_existing.reset_index().set_index("lemma"),
        how="outer",
        on="lemma",
        rsuffix="_r",
    )

    lemmas.loc[lemmas.index.notna(), "index"] = lemmas.index[lemmas.index.notna()]
    lemmas = lemmas.reset_index(drop=True).set_index("index").rename_axis(index=None)

    has_lemma_correct_cond = lemmas["lemma_correct"].notna()
    lemmas = pd.concat(
        [
            lemmas[has_lemma_correct_cond],
            lemmas[~has_lemma_correct_cond],
        ]
    )

    lemmas.to_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|")


update_word_lists()

# %%


def copy_lemmas_from_words_not_lemmas():
    words_not_lemmas = pd.read_csv(PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|", index_col=0)

    words = pd.DataFrame(words_not_lemmas["lemma"].map(mk_word, na_action="ignore"))
    lemmas_new = words[mk_is_lemma_cond(words["lemma"])]
    lemmas_existing = pd.read_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|", index_col=0)

    lemmas = pd.concat([lemmas_existing, lemmas_new])
    lemmas.sort_index(inplace=True)
    lemmas = lemmas[~lemmas["lemma"].isna()]
    lemmas = lemmas[~lemmas.duplicated("lemma")]

    lemmas.to_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|")


copy_lemmas_from_words_not_lemmas()

# %%


def update_lemmas_correct():
    # copy fixed lemmas
    words_not_lemmas = pd.read_csv(PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|", index_col=0)

    words = pd.DataFrame(words_not_lemmas["lemma"].map(mk_word, na_action="ignore"))
    lemmas_new = words[mk_is_lemma_cond(words["lemma"])]

    lemmas_existing = pd.read_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|", index_col=0)

    lemmas = pd.concat([lemmas_existing, lemmas_new])
    lemmas.sort_index(inplace=True)
    lemmas = lemmas[~lemmas["lemma"].isna()]
    lemmas = lemmas[~lemmas.duplicated("lemma")]

    nouns = lemmas[
        lemmas.apply(
            lambda x: float(int(x.name)) == x.name
            and is_noun(str(x["lemma"]))
            and is_noun(str(x["lemma_correct"])),
            axis=1,
        )
    ]
    noun_articles = get_lemmas_articles(nouns)
    nouns = nouns.join(noun_articles, rsuffix="_r")

    nouns.drop(columns=["lemma_r"], inplace=True)

    has_articles_cond = ~noun_articles["articles"].isna()
    nouns.loc[has_articles_cond, "articles"] = nouns.loc[
        has_articles_cond, "articles"
    ].map(lambda x: x.split(DEWIKI_ARTICLES_SEP))
    nouns = nouns.explode("articles")

    new_index = []
    counts = {}
    for idx in nouns.index:
        if idx not in counts:
            counts[idx] = 0
            new_index.append(idx)
        else:
            counts[idx] += 1
            new_index.append(idx + INDEX_SUFFIX.ALTERNATIVE_MEANING * counts[idx])

    nouns.index = new_index

    nouns["lemma_correct"] = nouns.apply(
        lambda x: (
            x["articles"]
            if pd.isna(x["articles"])
            else f"{ARTICLES_DICT[x["articles"]]} {mk_word(x["lemma"])}"
        ),
        axis=1,
    )

    nouns.drop(columns=["articles"], inplace=True)

    lemmas = lemmas.join(nouns, how="outer", rsuffix="_r").sort_index()

    has_lemma_cond = lemmas["lemma_r"].notna()
    lemmas.loc[has_lemma_cond, ["lemma", "lemma_correct"]] = lemmas.loc[
        has_lemma_cond, ["lemma_r", "lemma_correct_r"]
    ].values

    lemmas.drop(columns=["lemma_r", "lemma_correct_r"], inplace=True)

    no_lemma_correct_cond = lemmas["lemma_correct"].isna()
    lemmas.loc[no_lemma_correct_cond, "lemma_correct"] = lemmas.loc[
        no_lemma_correct_cond, "lemma"
    ]

    lemmas.to_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|")


update_lemmas_correct()

# %%


def partition_deck_for_generation():
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)

    has_data_cond = deck["part_of_speech"].notna()
    has_data = deck[has_data_cond].sort_index()
    has_no_data = deck[~has_data_cond].sort_index()

    is_correct_sentence_length_cond = has_data["sentence_de"].map(
        lambda x: SENTENCE_LENGTH.MIN <= len(x) <= SENTENCE_LENGTH.MAX
    )

    has_correct_sentence = has_data[is_correct_sentence_length_cond]
    has_incorrect_sentence = has_data[~is_correct_sentence_length_cond]
    has_incorrect_sentence = has_incorrect_sentence["word_de"]

    has_no_data = pd.concat([has_incorrect_sentence, has_no_data]).sort_index()

    deck = pd.concat([has_correct_sentence, has_no_data])

    deck.to_csv(PATH.DECK, sep="|")

    remove_separators(path=PATH.DECK, n=5)


partition_deck_for_generation()

# %%


def update_deck():
    lemmas = pd.read_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|", index_col=0)
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)

    words_de = pd.DataFrame(lemmas["lemma_correct"]).rename(
        columns={"lemma_correct": "word_de"}
    )

    deck = words_de.join(
        deck.reset_index().set_index("word_de"), how="outer", on="word_de"
    ).sort_index()

    deck.loc[deck["index"].isna(), "index"] = deck.index[deck["index"].isna()]
    deck = deck.reset_index(drop=True).set_index("index", drop=True)
    deck.rename_axis(index=None, inplace=True)
    deck = deck[deck.index.notna()]

    deck = deck[~deck.index.duplicated()].sort_index()
    deck = deck[~deck["word_de"].duplicated()]

    new_index = []
    for idx in deck.index:
        idx_rounded = float(int(idx))
        idx_suffix = round(idx % 1, ndigits=4)
        if idx_suffix >= INDEX_SUFFIX.ALTERNATIVE_MEANING and (
            idx_rounded not in deck.index
            or mk_word(deck.loc[idx, "word_de"])
            != mk_word(deck.loc[idx_rounded, "word_de"])
        ):
            continue
        new_index.append(idx)

    deck = deck.loc[new_index]

    deck.to_csv(PATH.DECK, sep="|")

    partition_deck_for_generation()


update_deck()

# %%


def update_dewiki_articles_dictionary():
    # The list can be downloaded here https://github.com/deemp/german-nouns/blob/main/german_nouns/nouns.csv

    dewiki = pd.read_csv(
        "../../../german-nouns/german_nouns/nouns.csv", low_memory=False
    )

    genuses = ["genus", "genus 1", "genus 2", "genus 3", "genus 4"]

    dewiki = dewiki[["lemma"] + genuses]

    def find_all_articles(x: pd.DataFrame):
        unique_elements = pd.unique(x[x.notna()].values.ravel())
        unique_elements = unique_elements[~pd.isna(unique_elements)]
        return DEWIKI_ARTICLES_SEP.join(unique_elements)

    noun_articles = pd.DataFrame(
        dewiki.groupby("lemma")[genuses].apply(find_all_articles)
    )

    noun_articles.columns = ["articles"]

    noun_articles.to_csv("data/dewiki-noun-articles.csv", sep="|")


update_dewiki_articles_dictionary()

# %%


def analyze_word_stats():
    path = "data/not-in-deck.csv"
    deck = pd.read_csv(path, sep="|", index_col=0)

    word_count = pd.DataFrame(
        deck.loc[:1782.001, "sentence_lemmatized_de"]
        .map(lambda x: x.split(";"))
        .explode()
        .value_counts()
    )

    word_count["word_de"] = word_count.index

    word_count.to_csv("data/word-statistics.csv", sep="|", index=None)


analyze_word_stats()
