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

DATA_PATH = Path("data")
LYRICS_PATH = DATA_PATH / "lyrics"
LYRICS_LEMMATIZED_PATH = LYRICS_PATH / "lemmatized.csv"
LYRICS_TEXTS_PATH = LYRICS_PATH / "texts.yaml"
LYRICS_WORDS_LEMMAS_PATH = LYRICS_PATH / "words-lemmas.csv"
LYRICS_WORDS_NOT_LEMMAS_PATH = LYRICS_PATH / "words-not-lemmas.csv"
LYRICS_WORDS_PATH = LYRICS_PATH / "words.csv"

DECK_PATH = Path(".") / "deck.csv"


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


def get_lemmas_articles(nouns: pd.DataFrame):
    dewiki_noun_articles = pd.read_csv("data/dewiki-noun-articles.csv", sep="|")
    lemma = "lemma"
    lemmas_initial = pd.DataFrame(nouns[lemma].map(mk_word))
    lemmas_with_articles = lemmas_initial.join(
        other=dewiki_noun_articles.set_index(lemma), on=lemma
    )
    return lemmas_with_articles


lemmata = pd.read_csv("data/dwds_lemmata_2025-01-15.csv")

# %%

import spacy

nlp = spacy.load("de_dep_news_trf")

# %%


def get_lemmatized_lyrics():
    with open(LYRICS_TEXTS_PATH, mode="r", encoding="UTF-8") as f:
        lyrics = yaml.safe_load(f)

    df = pd.DataFrame(lyrics, columns=["author", "title", "text"])
    df["text"] = df["text"].map(
        lambda x: "".join(
            [
                y
                for y in x.replace("\n", " ")
                if y.isalpha() or y.isnumeric() or y in [" ", "-"]
            ]
        )
    )

    df["text"] = df["text"].map(
        lambda x: (
            x
            # not lemmatized
            if LYRICS_LEMMATIZED_SEP not in x
            else LYRICS_LEMMATIZED_SEP.join(
                [tok.lemma_ for tok in nlp(x) if tok.lemma_ != "--"]
            )
        )
    )

    df.to_csv(LYRICS_LEMMATIZED_PATH, sep="|")


get_lemmatized_lyrics()
# %%


def get_word_lists():
    lyrics_lemmatized = pd.read_csv(LYRICS_LEMMATIZED_PATH, sep="|", index_col=0)
    lyrics_lemmatized["text"] = lyrics_lemmatized["text"].map(
        lambda x: str(x).split(LYRICS_LEMMATIZED_SEP)
    )
    lyrics_lemmatized = lyrics_lemmatized.explode("text")
    lyrics_lemmatized = lyrics_lemmatized.rename(columns={"text": "word"})
    lyrics_lemmatized = lyrics_lemmatized[~lyrics_lemmatized["word"].duplicated()]
    lyrics_lemmatized = lyrics_lemmatized.reset_index(drop=True)

    lyrics_lemmatized.to_csv(LYRICS_WORDS_PATH, sep="|")

    is_lemma_cond = lyrics_lemmatized["word"].isin(lemmata["lemma"])

    words_not_lemmas = lyrics_lemmatized[~is_lemma_cond]
    words_not_lemmas_existing = pd.read_csv(
        LYRICS_WORDS_NOT_LEMMAS_PATH, sep="|", index_col=0
    )
    words_not_lemmas = words_not_lemmas.join(words_not_lemmas_existing, rsuffix="_r")
    words_not_lemmas.drop(columns=["word_r", "author", "title"], inplace=True)
    words_not_lemmas.sort_index(inplace=True)
    words_not_lemmas = words_not_lemmas[~words_not_lemmas["word"].duplicated()]
    words_not_lemmas.to_csv(LYRICS_WORDS_NOT_LEMMAS_PATH, sep="|")

    words_lemmas = lyrics_lemmatized[is_lemma_cond]
    words_lemmas = words_lemmas.rename(columns={"word": "lemma"})
    words_lemmas_existing = pd.read_csv(LYRICS_WORDS_LEMMAS_PATH, sep="|", index_col=0)
    words_lemmas = words_lemmas.join(words_lemmas_existing, how="outer", rsuffix="_r")
    words_lemmas.loc[~words_lemmas["lemma_r"].isna(), "lemma"] = words_lemmas.loc[
        ~words_lemmas["lemma_r"].isna(), "lemma_r"
    ]
    words_lemmas.drop(columns=["author", "title", "lemma_r"], inplace=True)
    words_lemmas["lemma"] = words_lemmas["lemma"]
    words_lemmas.sort_index(inplace=True)
    words_lemmas = words_lemmas[
        ~words_lemmas["lemma"].map(mk_word, na_action="ignore").duplicated()
    ]
    words_lemmas.to_csv(LYRICS_WORDS_LEMMAS_PATH, sep="|")


get_word_lists()

# %%


def copy_lemmas_from_words_not_lemmas():
    words_not_lemmas = pd.read_csv(LYRICS_WORDS_NOT_LEMMAS_PATH, sep="|", index_col=0)

    lemma_cond = (
        words_not_lemmas["lemma"]
        .map(mk_word, na_action="ignore")
        .isin(lemmata["lemma"])
    )

    lemmas = words_not_lemmas.loc[
        lemma_cond,
        ["lemma"],
    ]

    words_lemmas = pd.read_csv(LYRICS_WORDS_LEMMAS_PATH, sep="|", index_col=0)
    lemmas_combined = pd.concat([words_lemmas, lemmas])
    lemmas_combined.sort_index(inplace=True)
    lemmas_combined = lemmas_combined[~lemmas_combined["lemma"].isna()]
    lemmas_combined = lemmas_combined[
        ~lemmas_combined["lemma"].map(mk_word, na_action="ignore").duplicated()
    ]

    lemmas_combined.to_csv(LYRICS_WORDS_LEMMAS_PATH, sep="|")


copy_lemmas_from_words_not_lemmas()

# %%


def update_lemmas_correct():
    lyrics_words_lemmas = pd.read_csv(LYRICS_WORDS_LEMMAS_PATH, sep="|", index_col=0)
    lyrics_words_nouns = lyrics_words_lemmas[
        lyrics_words_lemmas.apply(
            lambda x: is_noun(str(x["lemma"])),
            axis=1,
        )
    ]
    lyrics_words_nouns = lyrics_words_nouns[
        lyrics_words_nouns.index.map(lambda x: float(int(x)) == x)
    ]
    lemmas_articles = get_lemmas_articles(lyrics_words_nouns)
    lyrics_words_nouns = lyrics_words_nouns.join(lemmas_articles, rsuffix="_r")
    lyrics_words_nouns.drop(columns=["lemma_r"], inplace=True)

    has_articles_cond = ~lemmas_articles["articles"].isna()
    lyrics_words_nouns.loc[has_articles_cond, "articles"] = lyrics_words_nouns.loc[
        has_articles_cond, "articles"
    ].map(lambda x: x.split(DEWIKI_ARTICLES_SEP))
    lyrics_words_nouns = lyrics_words_nouns.explode("articles")

    new_index = []
    counts = {}
    for idx in lyrics_words_nouns.index:
        if idx not in counts:
            counts[idx] = 0
            new_index.append(idx)
        else:
            counts[idx] += 1
            new_index.append(idx + 0.1 * counts[idx])

    lyrics_words_nouns.index = new_index

    lyrics_words_nouns["lemma_correct"] = lyrics_words_nouns.apply(
        lambda x: (
            x["articles"]
            if pd.isna(x["articles"])
            else f"{ARTICLES_DICT[x["articles"]]} {mk_word(x["lemma"])}"
        ),
        axis=1,
    )

    lyrics_words_nouns.drop(columns=["articles"], inplace=True)

    lyrics_words_lemmas = lyrics_words_lemmas.join(
        lyrics_words_nouns, how="outer", rsuffix="_r"
    )
    lyrics_words_lemmas.loc[lyrics_words_nouns.index, "lemma_correct"] = (
        lyrics_words_nouns["lemma_correct"]
    )

    has_lemma_cond = ~lyrics_words_lemmas["lemma_r"].isna()
    lyrics_words_lemmas.loc[has_lemma_cond, "lemma"] = lyrics_words_lemmas.loc[
        has_lemma_cond, "lemma_r"
    ]
    lyrics_words_lemmas.drop(columns=["lemma_r", "lemma_correct_r"], inplace=True)

    # allow writing custom lemma_correct
    no_lemma_correct_cond = lyrics_words_lemmas["lemma_correct"].isna()
    lyrics_words_lemmas.loc[no_lemma_correct_cond, "lemma_correct"] = (
        lyrics_words_lemmas.loc[no_lemma_correct_cond, "lemma"]
    )

    lyrics_words_lemmas.to_csv(LYRICS_WORDS_LEMMAS_PATH, sep="|")


update_lemmas_correct()

    # lyrics_list = "\n".join(lyrics_split)

    # with open("song_lyrics_list.md", mode="w", encoding="UTF-8") as f:
    #     f.write(lyrics_list)


get_word_list()


# %%


def update_word_list():
    df = pd.read_csv("song_lyrics_list.csv", sep="|")
    df.columns = ["raw", "word_de"]
    df = df.drop(columns="raw")
    df1 = df[~df.duplicated()]
    df2 = df1.sort_values(by="word_de")
    df3 = df2.reset_index(drop=True)
    df3.to_csv("song_lyrics_list.csv", sep="|")


update_word_list()

# %%


def add_separable_verbs_to_vocabulary():
    separable_verbs = pd.read_csv(
        "data/verbs-separable.csv", sep="|", header=None, index_col=None
    )

    separable_verbs.columns = ["word_de"]

    vocabulary = pd.read_csv("data/vocabulary.csv", sep="|", index_col=0)

    result1 = pd.concat([separable_verbs, vocabulary], ignore_index=True)
    result2 = result1[~result1["word_de"].duplicated()]
    result3 = result2.reset_index(drop=True)

    result3.to_csv("data/vocabulary.csv", sep="|")


add_separable_verbs_to_vocabulary()

# %%


def find_difference_with_frequency_deck():
    frequency_df = pd.read_csv(
        "../../frequency/de/de-en/deck.csv", sep="|", index_col=0
    )

    # word_stat_df = pd.read_csv(
    #     "../../frequency/de/de-en/data/word-statistics.csv", sep="|"
    # )

    vocabulary_df = pd.read_csv("data/not-in-deck.csv", sep="|")
    vocabulary_df.columns = ["word_de"]

    res1 = vocabulary_df[~vocabulary_df["word_de"].isin(frequency_df["word_de"])]
    # res2 = res1[
    #     ~res1["word_de"]
    #     .map(
    #         lambda x: str(x)
    #         .removeprefix("das ")
    #         .removeprefix("der ")
    #         .removeprefix("die ")
    #     )
    #     .isin(word_stat_df["word_de"])
    # ]

    res1.to_csv("data/not-in-deck.csv", sep="|", index=None, header=None)


find_difference_with_frequency_deck()

# %%


def update_not_in_deck():
    path = "data/not-in-deck.csv"
    not_in_deck = pd.read_csv(path, sep="|", index_col=0)
    cond = not_in_deck["sentence_de"].map(
        lambda x: 30 <= len(x) <= 50, na_action="ignore"
    )
    not_in_deck_ok = not_in_deck[cond].sort_index()
    not_in_deck_bad = not_in_deck[~cond].sort_index()
    not_in_deck_bad.loc[
        :, ["part_of_speech", "word_en", "sentence_de", "sentence_en"]
    ] = ""

    pd.concat([not_in_deck_ok, not_in_deck_bad]).to_csv(
        path,
        sep="|",
    )

    with open(path, "r") as f:
        content = f.read()

    content = content.replace("||||", "")

    with open(path, "w") as f:
        f.write(content)


update_not_in_deck()

# %%


def find_not_in_deck_prev_that_are_not_in_deck():
    not_in_deck_prev = pd.read_csv("data/not-in-deck-prev.csv", sep="|")
    not_in_deck_prev.columns = ["word_de"]
    not_in_deck = pd.read_csv("data/not-in-deck.csv", sep="|")
    not_in_deck.columns = ["word_de"]

    not_in_deck_bad = not_in_deck_prev[
        ~not_in_deck_prev["word_de"].isin(not_in_deck["word_de"])
    ]

    not_in_deck_bad.to_csv("data/not-in-deck-bad.csv", index=None)


find_not_in_deck_prev_that_are_not_in_deck()


# %%


def find_not_lemma():
    not_in_deck = pd.read_csv("data/not-in-deck.csv", sep="|")
    lemmatized = pd.DataFrame(
        not_in_deck["sentence_lemmatized_de"].map(lambda x: x.split(";")).explode()
    )
    lemmatized.columns = ["lemma"]

    lemmatized = lemmatized[~lemmatized["lemma"].duplicated()]
    lemmatized.reset_index(drop=True, inplace=True)

    not_lemma = lemmatized[~lemmatized["lemma"].map(mk_word).isin(lemmata["lemma"])]

    not_lemma.columns = ["not_lemma"]

    not_lemma.to_csv("data/not-lemma.csv", sep="|")


find_not_lemma()

# %%


def replace_all_with_lemma():
    not_in_deck = pd.read_csv("data/not-in-deck.csv", sep="|", index_col=0)
    not_lemma = pd.read_csv("data/not-lemma.csv", sep="|", index_col=0)

    def replace_with_lemma(word: str):
        result = not_lemma.loc[not_lemma["not_lemma"] == word, ["lemma"]]
        if not result.empty:
            return result.iloc[0].values[0]

        return word

    print(replace_with_lemma(""))
    not_in_deck["sentence_lemmatized_de"] = pd.DataFrame(
        not_in_deck["sentence_lemmatized_de"].map(
            lambda x: ";".join([replace_with_lemma(y) for y in x.split(";")])
        )
    )

    not_in_deck.to_csv("data/not-in-deck.csv", sep="|")


replace_all_with_lemma()

# %%


def select_still_not_lemma():
    not_lemma = pd.read_csv("data/not-lemma.csv", sep="|", index_col=0)

    still_not_lemma = not_lemma[
        ~not_lemma["lemma"]
        .map(
            lambda x: str(x)
            .removeprefix("das ")
            .removeprefix("der ")
            .removeprefix("die ")
        )
        .isin(lemmata["lemma"])
    ]

    still_not_lemma["not_lemma"].to_csv("data/still-not-lemma.csv", sep="|")

    display(still_not_lemma)


select_still_not_lemma()

# %%

import spacy

nlp = spacy.load("de_dep_news_trf")

# %%


def lemmatize_not_in_deck():
    path = "data/not-in-deck.csv"
    not_in_deck = pd.read_csv(path, sep="|", index_col=0)
    not_in_deck["sentence_lemmatized_de"] = not_in_deck["sentence_de"].map(
        lambda x: ";".join([t.lemma_ for t in nlp(x) if t.lemma_ != "--"])
    )
    not_in_deck.to_csv(path, sep="|")


lemmatize_not_in_deck()

# %%


def find_nouns():
    deck = pd.read_csv("data/not-in-deck.csv", sep="|", index_col=0)
    words = pd.DataFrame(
        deck["sentence_lemmatized_de"].map(lambda x: x.split(";")).explode()
    )
    words.columns = ["lemma"]
    custom_deck_nouns = words[words["lemma"].map(is_noun)].sort_index()
    custom_deck_nouns["lemma"] = custom_deck_nouns["lemma"].map(mk_word)
    custom_deck_nouns = custom_deck_nouns[~custom_deck_nouns["lemma"].duplicated()]
    custom_deck_nouns.reset_index(drop=True, inplace=True)

    custom_deck_bad_nouns = custom_deck_nouns[custom_deck_nouns["lemma"].isin(lemmata["lemma"])]
    
    display(custom_deck_nouns)

    # custom_deck_nouns.columns = ["lemma"]
    # # nouns_without_article["sentence_id"] = nouns_without_article.index
    # custom_deck_nouns.to_csv("data/nouns-without-article.csv", sep="|")


find_nouns()

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


def set_articles():
    path = "data/nouns-without-article.csv"
    nouns_without_article = pd.read_csv(path, sep="|", index_col=0)
    # nouns_without_article["lemma_correct"] = pd.NA
    # nouns_without_article = nouns_without_article.astype(
    #     {"lemma": str, "lemma_correct": str}
    # )
    noun_articles = get_noun_articles(nouns_without_article)
    noun_articles["articles"] = noun_articles["articles"].map(
        lambda x: (
            ""
            if pd.isna(x)
            else ARTICLES_SEP.join(
                [ARTICLES_DICT[y] for y in x.split(DEWIKI_ARTICLES_SEP)]
            )
        )
    )
    noun_articles["lemma_correct"] = noun_articles.apply(
        lambda x: f"{x['articles']} {mk_word(x['lemma'])}", axis=1
    )

    display(noun_articles[["sentence_id", "lemma", "lemma_correct"]])
    # display(noun_articles[:50])

    # nouns_with_articles = .map(
    #     lambda x: f"{x['full']} {mk_word(x['lemma'])}"
    # )

    # number_articles = noun_articles["articles"].map(
    #     lambda x: 0 if pd.isna(x) else len(x.split(";"))
    # )

    # single_article_cond = number_articles.map(lambda x: x == 1)
    # no_articles_cond = number_articles.map(lambda x: x == 0)
    # multiple_articles_cond = number_articles.map(lambda x: x > 1)

    # nouns_with_single_article = noun_articles[single_article_cond]
    # nouns_with_no_article = noun_articles[no_articles_cond]
    # nouns_with_multiple_articles = noun_articles[multiple_articles_cond]

    # lemma_correct = "lemma_correct"

    # nouns_with_single_article_lemma_correct = pd.DataFrame(
    #     nouns_with_single_article.join(other=article_mapping, on="articles").apply(
    #         lambda x: f"{x['full']} {mk_word(x['lemma'])}", axis=1
    #     ),
    #     columns=[lemma_correct],
    # )

    # nouns_without_article.loc[
    #     nouns_with_single_article_lemma_correct.index, lemma_correct
    # ] = nouns_with_single_article_lemma_correct[lemma_correct]

    # nouns_without_article.to_csv(path, sep="|")

    # nouns_with_no_article.to_csv("data/nouns-with-no-article.csv", sep="|")


set_articles()
# %%


def read_noun_articles():
    nouns_many_articles = pd.read_csv("data/noun-articles.csv", sep="|")
    display(nouns_many_articles)

    # unique_article_cond = nouns_with_single_row[genuses].notna().sum(axis=1) == 1
    # no_article_cond = nouns_with_single_row[genuses].notna().sum(axis=1) == 0

    # nouns_with_single_row_with_unique_article = nouns_with_single_row[
    #     unique_article_cond
    # ]
    # nouns_with_single_row_with_no_articles = nouns_with_single_row[no_article_cond]

    # nouns_with_single_row_with_multiple_articles = nouns_with_single_row[
    #     ~(no_article_cond | unique_article_cond)
    # ]

    # display(nouns_with_single_row_with_multiple_articles)
    # nouns_with_multiple_rows.
    # display(nouns_with_multiple_rows)
    # nouns_with_multiple_articles = nouns_with_multiple_rows[]

    # display(nouns_with_single_row_with_unique_article)
    # display(nouns_with_single_row_with_multiple_articles)
    # display(nouns_with_single_row_with_no_articles)

    # nouns_with_multiple_rows

    # no_article = nouns_with_multiple_rows[
    #     nouns_with_multiple_rows[genuses].isna().all(axis=1)
    # ]
    # no_article = no_article[~no_article["lemma"].duplicated()]
    # display(no_article[["lemma"]])

    # display(unique_rows.sort_values(by="lemma"))

    # article_mapping = pd.DataFrame.from_records(
    #     columns=["short", "full"],
    #     data=[("f", "die"), ("m", "der"), ("n", "das")],
    # )

    # display(article_mapping)

    # article_mapping = {"f": "die", "m": "der", "n": "das"}

    # def get_article(word: str):
    #     try:
    #         results = dewiki.loc[dewiki["lemma"] == word, ["genus"]]
    #         results.dropna("genus", inplace=True)
    #         article = results.values[0]
    #         return article_mapping[article]
    #     except:
    #         display(dewiki.loc[dewiki["lemma"] == word, ["genus"]])
    #         print(word)
    #         return ""

    # # print(get_article("Sozialverhalten"))

    # articles = nouns_without_article.loc[:, "word"].map(get_article)
    # nouns_without_article["article"] = articles
    # nouns_without_article.columns = ["article", "word"]

    # nouns_without_article.to_csv(path, sep="|")


find_articles()
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
