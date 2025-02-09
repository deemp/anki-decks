# %%

import os
from pathlib import Path
import pandas as pd
from IPython.display import display
import yaml
import spacy
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import math

os.chdir(f'{os.environ["ROOT_DIR"]}/custom/de')
nlp = spacy.load("de_dep_news_trf")

# %%

ARTICLES_SHORT = ["f", "m", "n"]
ARTICLES_FULL = ["die", "der", "das"]
ARTICLES_SEP = "/"
DEWIKI_ARTICLES_SEP = ";"
LYRICS_LEMMATIZED_SEP = ";"
ARTICLES_DICT = dict(zip(ARTICLES_SHORT, ARTICLES_FULL))
DEWIKI_NOUN_ARTICLES = pd.read_csv("data/dewiki-noun-articles.csv", sep="|")
LEMMATA = pd.read_csv("data/dwds_lemmata_2025-01-15.csv")
MAX_OCCURENCES = 5


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
    WORD_COUNT = DATA / "word-count.csv"


class INDEX_SUFFIX:
    ALTERNATIVE_MEANING = 0.001
    NEW_WORD = 0.0001


class SENTENCE_LENGTH:
    MIN = 40
    MAX = 60


def make_baseform(word: str) -> str:
    # article
    if word in ARTICLES_FULL:
        return word
    # noun or name
    if word[0].isupper():
        return word
    if word[:3] in ARTICLES_FULL and word[3] == " ":
        k = 0
        for i, j in enumerate(word):
            k = i
            if j.isupper():
                break
        return word[k:]
    # anything else
    return word


def is_noun(word: str):
    return make_baseform(word)[0].isupper()


def get_lemmas_articles(nouns: pd.DataFrame):
    lemma = "lemma"
    lemmas_initial = pd.DataFrame(nouns[lemma].map(make_baseform))
    lemmas_with_articles = lemmas_initial.join(
        other=DEWIKI_NOUN_ARTICLES.set_index(lemma), on=lemma
    )
    return lemmas_with_articles


def make_is_lemma_cond(df: pd.DataFrame):
    return df.isin(LEMMATA["lemma"]) | df.isin(DEWIKI_NOUN_ARTICLES["lemma"])


def remove_separators_in_file(path: Path, n: int):
    with open(path, "r", encoding="UTF-8") as d:
        deck_text = d.read()

    with open(path, "w", encoding="UTF-8") as d:
        d.write(deck_text.replace("|" * n, ""))


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

    return LYRICS_LEMMATIZED_SEP.join(tokens)


def update_lemmatized_sources():
    sources_lemmatized = pd.read_csv(PATH.SOURCES_LEMMATIZED, sep="|", index_col=0)

    with open(PATH.SOURCES_TEXTS, mode="r", encoding="UTF-8") as f:
        sources = yaml.safe_load(f)

    sources_new = pd.DataFrame(
        sources,
        columns=["author", "title", "text"],
    ).astype(pd.StringDtype())

    sources = sources_new.merge(
        right=sources_lemmatized,
        on=["author", "title"],
        how="left",
    )

    is_lemmatized_right_cond = sources["text_y"].notna()
    sources.loc[is_lemmatized_right_cond, "text_x"] = sources.loc[
        is_lemmatized_right_cond, "text_y"
    ]
    sources = sources[["author", "title", "text_x"]].rename(columns={"text_x": "text"})

    sources_new_not_lemmatized = pd.DataFrame(sources[~is_lemmatized_right_cond])

    sources_new_not_lemmatized["text"] = sources_new_not_lemmatized["text"].map(
        lambda x: "".join(
            [
                y
                for y in x.replace("\n", " ")
                if y.isalpha() or y.isnumeric() or y in [" ", "-"]
            ]
        )
    )

    sources.loc[sources_new_not_lemmatized.index, "text"] = sources_new_not_lemmatized[
        "text"
    ].map(tokenize_sentence)

    sources.to_csv(PATH.SOURCES_LEMMATIZED, sep="|")


def update_sources_words() -> pd.DataFrame():
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

    return pd.DataFrame(words.loc[~words.duplicated("word"), ["word"]])


def update_words_not_lemmas(words_not_lemmas_new: pd.DataFrame()):
    if not Path(PATH.SOURCES_WORDS_NOT_LEMMAS).is_file():
        pd.DataFrame(columns=["word", "lemma", "lemma_correct"]).to_csv(
            PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|"
        )

    words_not_lemmas_existing = pd.read_csv(
        PATH.SOURCES_WORDS_NOT_LEMMAS,
        sep="|",
        index_col=0,
        dtype={"word": str, "lemma": str, "lemma_correct": str},
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

    has_lemma_correct_cond = words_not_lemmas["lemma_correct"].notna()
    words_not_lemmas.loc[~has_lemma_correct_cond, "lemma_correct"] = (
        words_not_lemmas.loc[~has_lemma_correct_cond, "lemma"]
    )

    words_not_lemmas.to_csv(PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|")

    return words_not_lemmas


# words: Puh
# words-not-lemmas: Puh|puh
# words-lemmas: puh|Puh
# puh comes from words-not-lemmas because it isn't a lemma
# hence, it's not available after join of words and words-lemmas

# Allow custom lemmas in a separate file
# words-not-lemmas-whitelist.csv
# list only lemmas there
# take indices from words-not-lemmas


def copy_lemmas_from_words_not_lemmas_to_words_lemmas(
    words_lemmas_new: pd.DataFrame(), words_not_lemmas: pd.DataFrame()
):
    words_lemmas_existing = pd.read_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|", index_col=0)

    words_lemmas_new = words_lemmas_new.join(
        words_lemmas_existing.reset_index(drop=True).set_index("lemma"), on="lemma"
    )
    words_lemmas_new.index = words_lemmas_new.index.map(float)

    lemmas_from_words_not_lemmas = (
        pd.DataFrame(
            words_not_lemmas[["lemma", "lemma_correct"]],
        )
        .dropna()
        .astype({"lemma": str, "lemma_correct": str})
    )

    lemmas_from_words_not_lemmas["lemma"] = lemmas_from_words_not_lemmas["lemma"].map(
        make_baseform, na_action="ignore"
    )

    lemmas_from_words_not_lemmas = lemmas_from_words_not_lemmas[
        make_is_lemma_cond(lemmas_from_words_not_lemmas["lemma"])
    ]

    words_lemmas = pd.concat(
        [words_lemmas_new, lemmas_from_words_not_lemmas]
    ).sort_index()

    words_lemmas = words_lemmas[
        ~words_lemmas["lemma_correct"]
        .map(make_baseform, na_action="ignore")
        .duplicated()
    ]

    words_lemmas.to_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|")

    return words_lemmas


def update_lemmas_correct(words_lemmas: pd.DataFrame) -> pd.DataFrame:
    nouns = words_lemmas[
        words_lemmas.apply(
            lambda x: float(int(x.name)) == x.name
            and is_noun(str(x["lemma"]))
            and (pd.isna(x["lemma_correct"]) or is_noun(str(x["lemma_correct"]))),
            axis=1,
        )
    ]

    nouns_articles = get_lemmas_articles(nouns)
    nouns = nouns.join(nouns_articles, rsuffix="_r")

    nouns.drop(columns=["lemma_r"], inplace=True)

    has_articles_cond = ~nouns_articles["articles"].isna()
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
            else f"{ARTICLES_DICT[x["articles"]]} {make_baseform(x["lemma"])}"
        ),
        axis=1,
    )

    nouns.drop(columns=["articles"], inplace=True)

    words_lemmas = words_lemmas.join(nouns, how="outer", rsuffix="_r").sort_index()

    has_lemma_cond = words_lemmas["lemma_correct_r"].notna()
    words_lemmas.loc[has_lemma_cond, ["lemma", "lemma_correct"]] = words_lemmas.loc[
        has_lemma_cond, ["lemma_r", "lemma_correct_r"]
    ].values

    words_lemmas.drop(columns=["lemma_r", "lemma_correct_r"], inplace=True)

    words_lemmas.to_csv(PATH.SOURCES_WORDS_LEMMAS, sep="|")

    return words_lemmas


def write_deck(deck: pd.DataFrame()):
    deck.to_csv(PATH.DECK, sep="|")
    remove_separators_in_file(path=PATH.DECK, n=5)


def copy_words_lemmas_to_deck(words_lemmas: pd.DataFrame()):
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)

    words_de = pd.DataFrame(words_lemmas["lemma_correct"]).rename(
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
            or make_baseform(deck.loc[idx, "word_de"])
            != make_baseform(deck.loc[idx_rounded, "word_de"])
        ):
            continue
        new_index.append(idx)

    deck = deck.loc[new_index]

    write_deck(deck=deck)

    return deck


def update_deck_lemmatized_sentences(deck: pd.DataFrame()):
    not_lemmatized_cond = deck["sentence_lemmatized_de"].isna()

    deck.loc[not_lemmatized_cond, "sentence_lemmatized_de"] = deck.loc[
        not_lemmatized_cond, "sentence_de"
    ].map(tokenize_sentence, na_action="ignore")

    write_deck(deck=deck)

    return deck


def update_word_counts(deck=pd.DataFrame()):
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)
    words = pd.DataFrame(
        deck["sentence_lemmatized_de"]
        .map(lambda x: x.split(";"), na_action="ignore")
        .explode("sentence_lemmatized_de")
    ).rename(columns={"sentence_lemmatized_de": "word_de"})

    word_counts = pd.DataFrame(words.value_counts())
    word_counts.to_csv(PATH.WORD_COUNT, sep="|")

    return word_counts


def check_is_correct_sentence(row: pd.Series(), word_stats: pd.DataFrame()) -> bool:
    word_de = make_baseform(row["word_de"])

    sentence_lemmatized_de = row["sentence_lemmatized_de"].split(LYRICS_LEMMATIZED_SEP)

    word_is_in_the_sentence = word_de in sentence_lemmatized_de

    words = pd.DataFrame(data=sentence_lemmatized_de, columns=["word_de"])

    word_counts = words.join(word_stats, on="word_de", rsuffix="r")

    # run this check on all sentences
    # because the number of sentences will grow
    # and the word counts will grow too
    sentence_contains_rare_words = any(
        (word_counts["word_de"] != word_de) & (word_counts["count"] <= MAX_OCCURENCES)
    )

    result = word_is_in_the_sentence & sentence_contains_rare_words

    # I subtract to correctly analyze other rows
    if not result:
        for word in words["word_de"]:
            word_stats.loc[word, "count"] -= 1

    return result


def partition_deck_by_having_data(deck: pd.DataFrame()):
    has_data_cond = deck["part_of_speech"].notna()
    rows_with_data = deck[has_data_cond].sort_index()
    rows_without_data = deck[~has_data_cond].sort_index()

    deck = pd.concat([rows_with_data, rows_without_data])

    write_deck(deck=deck)

    return rows_with_data, rows_without_data


def filter_deck_by_sentence_length(deck: pd.DataFrame()):
    rows_with_data, rows_without_data = partition_deck_by_having_data(deck=deck)

    is_good_sentence_length_cond = rows_with_data["sentence_de"].map(
        lambda x: SENTENCE_LENGTH.MIN <= len(x) <= SENTENCE_LENGTH.MAX
    )
    rows_with_good_sentence_length = rows_with_data[is_good_sentence_length_cond]
    rows_with_bad_sentence_length = pd.DataFrame(
        rows_with_data.loc[
            ~is_good_sentence_length_cond, ["word_de", "part_of_speech", "word_en"]
        ]
    )

    rows_without_data = pd.concat(
        [rows_with_bad_sentence_length, rows_without_data]
    ).sort_index()

    deck = pd.concat([rows_with_good_sentence_length, rows_without_data])

    write_deck(deck=deck)

    return deck


def partition_deck_for_generation(deck: pd.DataFrame()):
    deck = filter_deck_by_sentence_length(deck=deck)
    deck = update_deck_lemmatized_sentences(deck=deck)
    word_stats = update_word_counts(deck=deck)
    rows_with_data, rows_without_data = partition_deck_by_having_data(deck=deck)

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
            lambda row: check_is_correct_sentence(row=row, word_stats=word_stats),
            axis=1,
        )
        .iloc[::-1]
    )

    has_correct_sentence = rows_with_data[is_correct_sentence_cond]
    has_incorrect_sentence = rows_with_data[~is_correct_sentence_cond]
    has_incorrect_sentence = has_incorrect_sentence["word_de"]

    rows_without_data = pd.concat(
        [has_incorrect_sentence, rows_without_data]
    ).sort_index()

    deck = pd.concat([has_correct_sentence, rows_without_data])

    write_deck(deck=deck)

    return deck


def update_deck():
    deck = pd.read_csv(PATH.DECK, sep="|", index_col=0)
    partition_deck_for_generation(deck=deck)


def update_word_lists():
    # Fix overwrites some custom values ("puh")

    words = update_sources_words()
    words_not_lemmas = update_words_not_lemmas(
        words_not_lemmas_new=words[~make_is_lemma_cond(words["word"])]
    )
    words_lemmas = copy_lemmas_from_words_not_lemmas_to_words_lemmas(
        words_lemmas_new=words[make_is_lemma_cond(words["word"])].rename(
            columns={"word": "lemma"}
        ),
        words_not_lemmas=words_not_lemmas,
    )
    words_lemmas = update_lemmas_correct(words_lemmas=words_lemmas)

    deck = copy_words_lemmas_to_deck(words_lemmas=words_lemmas)

    filter_deck_by_sentence_length(deck=deck)


def update_all():
    update_word_lists()
    update_deck()


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


# %%

update_lemmatized_sources()

# %%

update_word_lists()

# %%

update_deck()

# %%

update_all()

# %%

update_dewiki_articles_dictionary()

# %%

update_word_counts()

# %%

from typing import Type

# Genius API access token (replace this with your own token)
ACCESS_TOKEN = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
BASE_URL = "https://api.genius.com"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def read_json(response):
    return response.json()


def read_text(response):
    return response.text()


async def get(
    session: type[aiohttp.ClientSession],
    url: str,
    params={},
    headers={},
    read_body=read_text,
):
    try:
        async with session.get(url=url, params=params, headers=headers) as response:
            resp = await read_body(response)
        return resp
    except Exception as e:
        print("Unable to get url {} due to {}.".format(url, e))
    return None


async def get_song_info(session: type[aiohttp.ClientSession], title: str, author: str):
    # Step 1: Search for the song on Genius using song title and artist name
    search_url = f"{BASE_URL}/search"

    params = {"q": f"{title} {author}"}

    response = await get(
        session=session,
        url=search_url,
        params=params,
        headers=headers,
        read_body=read_json,
    )

    if not response:
        return None, None

    search_results = response

    if search_results["response"]["hits"]:
        # Get the first song result
        song_hit = search_results["response"]["hits"][0]["result"]
        song_url = song_hit["url"]
        song_id = song_hit["id"]
        return song_url, song_id

    return None, None


async def get_lyrics(session: type[aiohttp.ClientSession], song_url: str):
    # Step 3: Scrape the lyrics from the Genius song page using BeautifulSoup
    response = await get(session=session, url=song_url, read_body=read_text)

    if not response:
        return None

    soup = BeautifulSoup(response, "html.parser")

    # Find the lyrics section
    lyrics = soup.find_all("div", {"data-lyrics-container": True})

    if lyrics:
        return "\n\n".join([x.get_text(separator="\n").strip() for x in lyrics])

    return None


async def get_lyrics_by_title(
    session: type[aiohttp.ClientSession], title: str, author: str
):
    song_url, _ = await get_song_info(session=session, title=title, author=author)

    if song_url:
        lyrics = await get_lyrics(session=session, song_url=song_url)
        if lyrics:
            print(f"Found lyrics: {title} by {author}")
            return lyrics
        print(f"Lyrics not found: {title} by {author}")
    else:
        print(f"No info found: {title} by {author}")
    return None


async def gather_concurrently(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def get_texts(df: type[pd.DataFrame], path: str):
    block_size = 10
    df_na = df[df["lyrics"].isna()]
    block_count = math.ceil(df_na.shape[0] / block_size)

    for i in range(block_count):
        print(f"{i=}")
        block_df = df_na.iloc[i * block_size : (i + 1) * block_size]
        titles = block_df[["title", "author"]].to_numpy().tolist()

        async with aiohttp.ClientSession(headers=headers) as session:
            texts = await gather_concurrently(
                block_size,
                *(
                    get_lyrics_by_title(session=session, title=title, author=author)
                    for title, author in titles
                ),
            )

        for idx, text in zip(block_df.index.tolist(), texts):
            df.loc[idx, "lyrics"] = repr(text)

        df.to_csv(path, sep="|")


async def update_songs():
    path = "data/song-titles.csv"
    df = pd.read_csv(path, sep="|", index_col=0)
    await get_texts(df=df, path=path)

    df = pd.read_csv(path, sep="|", index_col=0)
    has_lyrics_cond = df["lyrics"].notna()
    df = (
        pd.concat([df[has_lyrics_cond], df[~has_lyrics_cond]])
        .reindex()
        .reset_index(drop=True)
    )
    df.to_csv(path, sep="|")


# %%

await update_songs()
