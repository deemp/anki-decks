# %%

import os
from pathlib import Path
import asyncio
import math
import textwrap
import importlib
import pandas as pd
from IPython.display import display
import spacy
from bs4 import BeautifulSoup
import aiohttp
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

os.chdir(f'{os.environ["ROOT_DIR"]}/custom/de')

yaml = YAML()

nlp = spacy.load("de_dep_news_trf")

# %%

import custom.de.script.lib as lib

importlib.reload(lib)

from custom.de.script.lib import (
    DEWIKI_ARTICLES_SEP,
    ARTICLES_DICT,
    DEWIKI_NOUN_ARTICLES,
    LEMMATA,
    make_baseform,
    tokenize_sentence,
    remove_separators_in_file,
    make_is_lemma_cond,
    APIRequestsArgs,
    update_deck_raw,
    MODEL,
    read_csv,
    update_words_bad_baseform,
)


LYRICS_LEMMATIZED_SEP = ";"
MAX_OCCURENCES = 2


class PATH:
    DATA = Path("data")
    PWD = Path(".")
    DECK = PWD / "deck.csv"
    SOURCES = DATA / "sources"
    PLAYLIST = SOURCES / "playlist"
    WORDS = SOURCES / "words"
    KNOWN = WORDS / "known.csv"
    SOURCES_LEMMATIZED = SOURCES / "lemmatized.csv"
    PLAYLIST_DATA = PLAYLIST / "data.csv"
    PLAYLIST_DATA_YAML = PLAYLIST / "data.yaml"
    SOURCES_TEXTS = SOURCES / "data.yaml"
    SOURCES_WORDS_LEMMAS = SOURCES / "words-lemmas.csv"
    SOURCES_WORDS_NOT_LEMMAS = SOURCES / "words-not-lemmas.csv"
    SOURCES_WORDS = SOURCES / "words.csv"
    WORD_COUNT = DATA / "word-count.csv"
    PARALLEL_REQUESTS = DATA / "parallel-requests.jsonl"
    PARALLEL_RESPONSES = DATA / "parallel-responses.jsonl"
    PARALLEL_RESPONSES_CONCATENATED = DATA / "parallel-responses-concatenated.csv"
    WORDS_BAD_BASEFORM = DATA / "words-bad-baseform.csv"


class INDEX_SUFFIX:
    ALTERNATIVE_MEANING = 0.001
    NEW_WORD = 0.0001


class SENTENCE_LENGTH:
    MIN = 60
    MAX = 70


class PART:
    ITERATIONS = 2
    SIZE = 10
    COUNT_PER_ITERATION = 2


# Genius API access token (replace this with your own token)
ACCESS_TOKEN = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
BASE_URL = "https://api.genius.com"

PLAYLIST_DATA_PATH = "data/sources/playlist/data.csv"
PLAYLIST_DATA_YAML_PATH = "data/sources/playlist/data.yaml"
PLAYLIST_RAW_PATH = "data/sources/playlist/raw.csv"


# TODO use the list of nouns and the list of lemmata (uppercase)
def is_noun(word: str):
    return make_baseform(word)[0].isupper()


def get_lemmas_articles(nouns: pd.DataFrame):
    lemma = "lemma"
    lemmas_initial = pd.DataFrame(nouns[lemma].map(make_baseform))
    lemmas_with_articles = lemmas_initial.join(
        other=DEWIKI_NOUN_ARTICLES.set_index(lemma), on=lemma
    )
    return lemmas_with_articles


def update_lemmatized_sources():
    sources_lemmatized = read_csv(PATH.SOURCES_LEMMATIZED)

    with open(PATH.PLAYLIST_DATA_YAML, mode="r", encoding="UTF-8") as f:
        sources = yaml.load(f)

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
        ),
        na_action="ignore",
    )

    sources.loc[sources_new_not_lemmatized.index, "text"] = sources_new_not_lemmatized[
        "text"
    ].map(tokenize_sentence, na_action="ignore")

    sources.to_csv(PATH.SOURCES_LEMMATIZED, sep="|")


def update_sources_words() -> pd.DataFrame():
    lyrics_lemmatized = read_csv(PATH.SOURCES_LEMMATIZED)

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
        pd.DataFrame(
            columns=["word", "lemma", "part_of_speech", "lemma_correct", "is_lemma"]
        ).to_csv(PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|")

    words_not_lemmas_existing = pd.read_csv(
        PATH.SOURCES_WORDS_NOT_LEMMAS,
        sep="|",
        index_col=0,
        dtype={
            "word": str,
            "part_of_speech": str,
            "lemma": str,
            "lemma_correct": str,
            "is_lemma": str,
        },
    )

    no_lemma_correct_cond = words_not_lemmas_existing["lemma_correct"].isna()
    words_not_lemmas_existing.loc[no_lemma_correct_cond, "lemma_correct"] = (
        words_not_lemmas_existing.loc[no_lemma_correct_cond, "lemma"]
    )

    words_not_lemmas = words_not_lemmas_new.join(
        other=words_not_lemmas_existing.set_index("word"), on="word"
    )

    words_not_lemmas = words_not_lemmas[~words_not_lemmas["word"].isin(["null", "nan"])]

    words_not_lemmas.sort_index(inplace=True)
    words_not_lemmas = words_not_lemmas[~words_not_lemmas["word"].duplicated()]

    has_lemma_correct_cond = words_not_lemmas["lemma_correct"].notna()
    words_not_lemmas.loc[~has_lemma_correct_cond, "lemma_correct"] = (
        words_not_lemmas.loc[~has_lemma_correct_cond, "lemma"]
    )

    is_lemma_cond = make_is_lemma_cond(words_not_lemmas["lemma_correct"])

    not_lemma = words_not_lemmas.loc[~is_lemma_cond]

    words_not_lemmas = pd.concat(
        [
            words_not_lemmas[is_lemma_cond],
            not_lemma[not_lemma["lemma"].notna()],
            not_lemma[not_lemma["lemma"].isna()],
        ]
    )

    words_not_lemmas.loc[is_lemma_cond, "is_lemma"] = True
    words_not_lemmas.loc[~is_lemma_cond, "is_lemma"] = False
    words_not_lemmas.loc[words_not_lemmas["lemma"].isna(), "is_lemma"] = None

    words_not_lemmas.to_csv(PATH.SOURCES_WORDS_NOT_LEMMAS, sep="|")

    return words_not_lemmas


def copy_lemmas_from_words_not_lemmas_to_words_lemmas(
    words_lemmas_new: pd.DataFrame(), words_not_lemmas: pd.DataFrame()
):
    words_lemmas_existing = read_csv(PATH.SOURCES_WORDS_LEMMAS)

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

    has_articles_cond = nouns["articles"].notna()

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
    remove_separators_in_file(path=PATH.DECK)


def copy_words_lemmas_to_deck(words_lemmas: pd.DataFrame()):
    deck = read_csv(PATH.DECK)

    deck_custom_rows = deck[
        deck.index.map(
            lambda x: 0
            < round(float(x) % 1, ndigits=4)
            < INDEX_SUFFIX.ALTERNATIVE_MEANING
        )
    ]

    known = read_csv(PATH.KNOWN)

    deck = deck.set_index("word_de")

    words_de = pd.DataFrame(words_lemmas["lemma_correct"]).rename(
        columns={"lemma_correct": "word_de"}
    )

    is_word_de_known_cond = words_de["word_de"].isin(known["word_de"])

    words_de_unknown = words_de[~is_word_de_known_cond]

    deck = words_de_unknown.join(
        deck,
        on="word_de",
    )

    deck = pd.concat([deck, deck_custom_rows]).sort_index()

    deck = deck[~deck.index.duplicated()]
    deck = deck[~deck["word_de"].duplicated()]
    deck = deck[~deck["word_de"].str.startswith("-")]

    deck = pd.concat(
        [
            deck[deck["word_en"].notna()].sort_index(),
            deck[deck["word_en"].isna()].sort_index(),
        ]
    )

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


def partition_deck_by_having_sentence(deck: pd.DataFrame()):
    has_data_cond = deck["sentence_de"].notna()
    rows_with_data = deck[has_data_cond].sort_index()
    rows_without_data = deck.loc[~has_data_cond, "word_de"].sort_index()

    deck = pd.concat([rows_with_data, rows_without_data])

    write_deck(deck=deck)

    return rows_with_data, rows_without_data


def filter_deck_by_sentence_length(deck: pd.DataFrame()):
    rows_with_sentence, rows_without_sentence = partition_deck_by_having_sentence(
        deck=deck
    )

    is_good_sentence_length_cond = rows_with_sentence["sentence_de"].map(
        lambda x: (
            SENTENCE_LENGTH.MIN <= len(x) <= SENTENCE_LENGTH.MAX
            if isinstance(x, str)
            else False
        )
    )
    rows_with_good_sentence_length = rows_with_sentence[is_good_sentence_length_cond]
    rows_with_bad_sentence_length = pd.DataFrame(
        rows_with_sentence.loc[~is_good_sentence_length_cond, ["word_de"]]
    )

    deck = pd.concat(
        [
            rows_with_good_sentence_length,
            rows_with_bad_sentence_length,
            rows_without_sentence,
        ]
    )

    write_deck(deck=deck)

    return deck


def partition_deck_for_generation(deck: pd.DataFrame()):
    deck = filter_deck_by_sentence_length(deck=deck)
    deck = update_deck_lemmatized_sentences(deck=deck)
    word_stats = update_word_counts(deck=deck)
    rows_with_sentence, rows_without_sentence = partition_deck_by_having_sentence(
        deck=deck
    )

    # prefer removing rows with a larger index
    # to change rows with smaller index less frequently
    # to preserve progress
    #
    # rows with smaller indices may get removed
    # if we leave rows with larger indices that introduced some words
    # that increased word counts
    # and hence disallowed rows with smaller indices to stay
    is_correct_sentence_cond = (
        rows_with_sentence.iloc[::-1]
        .apply(
            lambda row: check_is_correct_sentence(row=row, word_stats=word_stats),
            axis=1,
        )
        .iloc[::-1]
    )

    rows_with_correct_sentence = rows_with_sentence[is_correct_sentence_cond]

    rows_with_incorrect_sentence = rows_with_sentence.loc[~is_correct_sentence_cond]

    rows_without_sentence = pd.concat(
        [rows_with_incorrect_sentence, rows_without_sentence]
    ).sort_index()["word_de"]

    deck = pd.concat([rows_with_correct_sentence, rows_without_sentence])

    write_deck(deck=deck)

    return deck


def update_word_lists():
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

    deck = filter_deck_by_sentence_length(deck=deck)

    update_words_bad_baseform(
        deck_raw=deck, words_bad_baseform_path=PATH.WORDS_BAD_BASEFORM
    )


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


async def generate_deck_data():
    print("Starting update")

    api_requests_args = APIRequestsArgs(
        requests_filepath=f"{PATH.DATA}/parallel-requests.jsonl",
        save_filepath=f"{PATH.DATA}/parallel-responses.jsonl",
        max_attempts=3,
        request_url="https://api.openai.com/v1/chat/completions",
    )

    await update_deck_raw(
        word_counts_path=PATH.WORD_COUNT,
        deck_raw_path=PATH.DECK,
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
        has_part_of_speech=False,
        has_word_en=False,
    )

    print("Update completed!")


async def update_deck_data():
    for i in range(PART.ITERATIONS):
        print(f"Iteration: {i}")

        await generate_deck_data()


def get_response_json(response):
    return response.json()


def get_response_text(response):
    return response.text()


async def send_get_request(
    session: type[aiohttp.ClientSession],
    url: str,
    params=None,
    headers=None,
    read_body=get_response_text,
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

    search_results = await send_get_request(
        session=session,
        url=search_url,
        params=params,
        read_body=get_response_json,
    )

    if not search_results:
        return None, None

    if search_results["response"]["hits"]:
        # Get the first song result
        song_hit = search_results["response"]["hits"][0]["result"]
        song_url = song_hit["url"]
        song_id = song_hit["id"]
        return song_url, song_id

    return None, None


async def get_lyrics(session: type[aiohttp.ClientSession], song_url: str):
    # Step 3: Scrape the lyrics from the Genius song page using BeautifulSoup
    response = await send_get_request(
        session=session, url=song_url, read_body=get_response_text
    )

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


async def get_texts(df: type[pd.DataFrame], path: str, titles_no_lyrics: [str]):
    block_size = 10
    df_na = df[df["text"].isna()]
    block_count = math.ceil(df_na.shape[0] / block_size)
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    for i in range(block_count):
        print(f"{i=}")
        block_df = df_na.iloc[i * block_size : (i + 1) * block_size]
        title_authors = block_df[["title", "author"]].to_numpy().tolist()

        async with aiohttp.ClientSession(headers=headers) as session:
            texts = await gather_concurrently(
                block_size,
                *(
                    get_lyrics_by_title(session=session, title=title, author=author)
                    for title, author in title_authors
                ),
            )

        for idx, text, title_author in zip(
            block_df.index.tolist(), texts, title_authors
        ):
            title, author = title_author
            if df.loc[idx, "title"] not in titles_no_lyrics:
                df.loc[idx, "text"] = text
            else:
                print(f"Ignoring lyrics: {title} by {author}")

        df = df.replace(r"\n", r"\\n", regex=True)

        df.to_csv(path, sep="|")


def strip_texts(df: type[pd.DataFrame]):
    df.loc[:, "text"] = df.loc[:, "text"].map(lambda x: x.strip('"').strip())
    return df


def LS(s):
    return LiteralScalarString(textwrap.dedent(s))


def copy_texts_from_df_to_yaml(df: type[pd.DataFrame], path_yaml: str):
    data = []
    for idx in df.index:
        x = df.loc[idx]
        text = x["text"]
        if pd.notna(text):
            l = [line.rstrip() for line in text.split("\\n")]
            text = "\n".join(l)
            text = text.strip()
            text = LS(text)
        data.append({"title": x["title"], "author": x["author"], "text": text})

    with open(path_yaml, mode="w", encoding="UTF-8") as p:
        yaml.dump(data=data, stream=p)


def copy_texts_from_yaml_to_df(path_yaml: str):
    data = yaml.load(Path(path_yaml))
    for i, _ in enumerate(data):
        if pd.notna(data[i]["text"]):
            data[i]["text"] = data[i]["text"]
    return pd.DataFrame(data, columns=["title", "author", "text"])


async def update_songs():
    playlist_raw = read_csv(PLAYLIST_RAW_PATH)
    playlist_data = copy_texts_from_yaml_to_df(path_yaml=PLAYLIST_DATA_YAML_PATH)

    playlist_data = (
        playlist_raw.join(
            playlist_data.set_index(["title", "author"]),
            on=["title", "author"],
            rsuffix="_r",
        )
        .reset_index()
        .drop(columns=["index"])
    )

    titles_no_lyrics = [
        "Das Schaf",
        "##@@@ (zeig mir was neues)",
        "Böhmermann ist Schuld",
        "To Do Liste",
        "Hegendary",
    ]

    await get_texts(
        df=playlist_data, path=PLAYLIST_DATA_PATH, titles_no_lyrics=titles_no_lyrics
    )

    playlist_data = read_csv(PLAYLIST_DATA_PATH)

    has_text_cond = playlist_data["text"].notna()

    df_has_text = playlist_data[has_text_cond]
    df_has_no_text = playlist_data[~has_text_cond]

    df_has_text = strip_texts(df_has_text)

    df_no_newline_in_text = df_has_text[
        df_has_text["text"].map(lambda x: r"\n" not in x)
    ]

    for idx in df_no_newline_in_text.index:
        x = df_has_text.loc[idx]
        print(f"Bad formatting: {x.name}) {x["title"]} by {x["author"]}")

    playlist_data = (
        pd.concat([df_has_text, df_has_no_text]).reindex().reset_index(drop=True)
    )

    playlist_data.to_csv(PLAYLIST_DATA_PATH, sep="|")

    copy_texts_from_df_to_yaml(df=playlist_data, path_yaml=PLAYLIST_DATA_YAML_PATH)


# %%

await update_songs()
# %%

update_lemmatized_sources()

# %%

update_word_lists()

# %%

await update_deck_data()

# %%

update_dewiki_articles_dictionary()

# %%

update_word_counts()
