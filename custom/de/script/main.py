# %%

import os
from pathlib import Path
import asyncio
import math
import textwrap
import importlib
import pandas as pd
from typing import List
from IPython.display import display
import spacy
from bs4 import BeautifulSoup
import aiohttp
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from iso639 import Lang

os.chdir(f'{os.environ["ROOT_DIR"]}/custom/de')

yaml = YAML()

nlp = spacy.load("de_core_news_lg")

# %%

import custom.de.script.lib as lib

importlib.reload(lib)

from custom.de.script.lib import (
    ARTICLES_DICT,
    Model,
    ConstConfig,
    ConstPath,
    ConstRareWords,
    ConstGenerationSettings,
    ConstPromptSettings,
    ConstSentenceLength,
    ConstColumnNames,
    ConstLanguageSettings,
    make_baseform,
    tokenize_sentence,
    remove_separators_in_file,
    ApiRequestsArgs,
    update_deck_raw,
    read_csv,
    update_words_bad_baseform,
    generate_deck_data_iteratively,
    filter_deck_raw_by_sentence_length,
    update_word_counts,
)


DEWIKI_ARTICLES_SEP = ";"
LYRICS_LEMMATIZED_SEP = ";"


class PATH_ALL:
    data = Path("data")
    pwd = Path(".")
    deck = pwd / "deck.csv"
    sources = data / "sources"
    playlist = sources / "playlist"
    sources_lemmatized = sources / "lemmatized.csv"
    sources_words = sources / "words.csv"
    playlist_raw = playlist / "raw.csv"
    playlist_data = playlist / "data.csv"
    playlist_data_yaml = playlist / "data.yaml"
    words = data / "words"
    words_known = words / "known.csv"
    words_lemmas = words / "lemmas.csv"
    words_not_lemmas = words / "not-lemmas.csv"
    too_frequent = words / "too-frequent.csv"
    word_counts = words / "counts.csv"
    words_bad_baseform = words / "bad-baseform.csv"
    api = data / "api"
    parallel_requests = api / "parallel-requests.jsonl"
    parallel_responses = api / "parallel-responses.jsonl"
    parallel_responses_concatenated = api / "parallel-responses-concatenated.csv"
    external = data / "external"
    dewiki_nouns = external / "dewiki-nouns.csv"
    dewiki_noun_articles = external / "dewiki-noun-articles.csv"
    dwds_lemmata = external / "dwds_lemmata_2025-01-15.csv"


class INDEX_SUFFIX:
    ALTERNATIVE_MEANING = 0.001
    NEW_WORD = 0.0001


GENERATION_SETTINGS = ConstGenerationSettings(
    iterations=20, block_size=70, blocks_per_iteration=30
)

PATH = ConstPath(
    word_counts=PATH_ALL.word_counts,
    deck_raw=PATH_ALL.deck,
    words_bad_baseform=PATH_ALL.words_bad_baseform,
    parallel_requests=PATH_ALL.parallel_requests,
    parallel_responses=PATH_ALL.parallel_responses,
    parallel_responses_concatenated=PATH_ALL.parallel_responses_concatenated,
)

SENTENCE_LENGTH = ConstSentenceLength(mini=60, maxi=70)

RARE_WORDS = ConstRareWords(min_count_in_sentence=2, max_occurences_in_deck=3)

# Genius API access token (replace this with your own token)
ACCESS_TOKEN = os.getenv("GENIUS_CLIENT_ACCESS_TOKEN")
BASE_URL = "https://api.genius.com"

API_REQUESTS_ARGS = ApiRequestsArgs(
    requests_filepath=PATH_ALL.parallel_requests,
    save_filepath=PATH_ALL.parallel_responses,
    max_attempts=3,
    request_url="https://api.openai.com/v1/chat/completions",
)

PROMPT_SETTINGS = ConstPromptSettings(has_part_of_speech=False, has_word_en=False)

LEMMATIZED_SEP = ";"

LANGUAGE_SETTINGS = ConstLanguageSettings(lang_1=Lang("German"), lang_2=Lang("English"))

COLUMN_NAMES = ConstColumnNames.from_language_settings(LANGUAGE_SETTINGS)

CONFIG = ConstConfig(
    rare_words=RARE_WORDS,
    generation_settings=GENERATION_SETTINGS,
    sentence_length=SENTENCE_LENGTH,
    path=PATH,
    model=Model.CHATGPT_4O_MINI,
    prompt_settings=PROMPT_SETTINGS,
    nlp=nlp,
    api_requests_args=API_REQUESTS_ARGS,
    lemmatized_sep=LEMMATIZED_SEP,
    column_names=COLUMN_NAMES,
    language_settings=LANGUAGE_SETTINGS,
)

DEWIKI_NOUN_ARTICLES = pd.read_csv(PATH_ALL.dewiki_noun_articles, sep="|")
LEMMATA = pd.read_csv(PATH_ALL.dwds_lemmata)


def make_is_lemma_cond(s: type[pd.Series]):
    s = s.map(lambda x: make_baseform(x) if isinstance(x, str) else False)
    return s.isin(LEMMATA["lemma"]) | s.isin(DEWIKI_NOUN_ARTICLES["lemma"])


# TODO use the list of nouns and the list of lemmata (uppercase)
def is_noun(word: str):
    return make_baseform(word)[0].isupper()


def get_lemmas_articles(nouns: type[pd.DataFrame]):
    lemma = "lemma"
    lemmas_initial = pd.DataFrame(nouns[lemma].map(make_baseform))
    lemmas_with_articles = lemmas_initial.join(
        other=DEWIKI_NOUN_ARTICLES.set_index(lemma), on=lemma
    )
    return lemmas_with_articles


def update_lemmatized_sources(config: type[ConstConfig]):
    sources_lemmatized = read_csv(PATH_ALL.sources_lemmatized)

    with open(PATH_ALL.playlist_data_yaml, mode="r", encoding="UTF-8") as f:
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
    ].map(lambda x: tokenize_sentence(config=config, sentence=x), na_action="ignore")

    sources.to_csv(PATH_ALL.sources_lemmatized, sep="|")


def update_sources_words() -> type[pd.DataFrame]:
    lyrics_lemmatized = read_csv(PATH_ALL.sources_lemmatized)

    texts = lyrics_lemmatized[["text"]]
    texts.loc[:, "text"] = texts.loc[:, "text"].map(
        lambda x: str(x).split(LYRICS_LEMMATIZED_SEP)
    )
    texts = texts.rename(columns={"text": "word"})

    words = texts.explode("word")
    words.reset_index(names="song_id", inplace=True)
    words = words[["song_id", "word"]]
    words.to_csv(PATH_ALL.sources_words, sep="|")

    return pd.DataFrame(words.loc[~words.duplicated("word"), ["word"]])


def update_words_not_lemmas(words_not_lemmas_new: type[pd.DataFrame]):
    if not Path(PATH_ALL.words_not_lemmas).is_file():
        pd.DataFrame(
            columns=["word", "lemma", "part_of_speech", "lemma_correct", "is_lemma"]
        ).to_csv(PATH_ALL.words_not_lemmas, sep="|")

    words_not_lemmas_existing = pd.read_csv(
        PATH_ALL.words_not_lemmas,
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

    words_not_lemmas.to_csv(PATH_ALL.words_not_lemmas, sep="|")

    return words_not_lemmas


def copy_lemmas_from_words_not_lemmas_to_words_lemmas(
    words_lemmas_new: type[pd.DataFrame], words_not_lemmas: type[pd.DataFrame]
):
    words_lemmas_existing = read_csv(PATH_ALL.words_lemmas)

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

    words_lemmas.to_csv(PATH_ALL.words_lemmas, sep="|")

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
            else f"{ARTICLES_DICT[x['articles']]} {make_baseform(x['lemma'])}"
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

    words_lemmas.to_csv(PATH_ALL.words_lemmas, sep="|")

    return words_lemmas


def write_deck(deck: type[pd.DataFrame]):
    deck.to_csv(PATH_ALL.deck, sep="|")
    remove_separators_in_file(path=PATH_ALL.deck)


def copy_words_lemmas_to_deck(
    config: type[ConstConfig], words_lemmas: type[pd.DataFrame]
):
    deck = read_csv(PATH_ALL.deck)

    deck_custom_rows = deck[
        deck.index.map(
            lambda x: 0
            < round(float(x) % 1, ndigits=4)
            < INDEX_SUFFIX.ALTERNATIVE_MEANING
        )
    ]

    known = read_csv(PATH_ALL.words_known)

    deck = deck.set_index(config.column_names.word_lang_1)

    words_de = pd.DataFrame(words_lemmas["lemma_correct"]).rename(
        columns={"lemma_correct": config.column_names.word_lang_1}
    )

    is_word_de_known_cond = words_de[config.column_names.word_lang_1].isin(
        known[config.column_names.word_lang_1]
    )

    words_de_unknown = words_de[~is_word_de_known_cond]

    deck = words_de_unknown.join(
        deck,
        on=config.column_names.word_lang_1,
    )

    deck = pd.concat([deck, deck_custom_rows]).sort_index()

    deck = deck[~deck.index.duplicated()]
    deck = deck[~deck[config.column_names.word_lang_1].duplicated()]
    deck = deck[~deck[config.column_names.word_lang_1].str.startswith("-")]

    deck = pd.concat(
        [
            deck[deck[config.column_names.word_lang_2].notna()].sort_index(),
            deck[deck[config.column_names.word_lang_2].isna()].sort_index(),
        ]
    )

    write_deck(deck=deck)

    return deck


def update_word_lists(config: type[ConstConfig]):
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

    deck = copy_words_lemmas_to_deck(config=config, words_lemmas=words_lemmas)

    deck = filter_deck_raw_by_sentence_length(config=config, deck_raw=deck)

    update_words_bad_baseform(config=config, deck_raw=deck)


def update_dewiki_articles_dictionary():
    # The list can be downloaded here https://github.com/deemp/german-nouns/blob/main/german_nouns/nouns.csv

    dewiki = pd.read_csv(PATH_ALL.dewiki_nouns, low_memory=False)

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

    noun_articles.to_csv(PATH_ALL.dewiki_noun_articles, sep="|")


async def generate_deck_data(config: type[ConstConfig]):
    print("Starting update")

    update_word_lists(config=config)

    await update_deck_raw(config=CONFIG)

    print("Update completed!")


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


async def get_texts(df: type[pd.DataFrame], path: str, titles_no_lyrics: List[str]):
    block_size = 10
    df_na = df[df["text"].isna()]
    block_count = math.ceil(df_na.shape[0] / block_size)
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    for i in range(block_count):
        print(f"Songs block: {i}")
        block_df = df_na.iloc[i * block_size : (i + 1) * block_size]

        display(block_df[["title", "author"]])

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
    playlist_raw = pd.read_csv(PATH_ALL.playlist_raw, sep="|", index_col=None)
    playlist_data = copy_texts_from_yaml_to_df(path_yaml=PATH_ALL.playlist_data_yaml)

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
        df=playlist_data,
        path=PATH_ALL.playlist_data,
        titles_no_lyrics=titles_no_lyrics,
    )

    playlist_data = read_csv(PATH_ALL.playlist_data)

    has_text_cond = playlist_data["text"].notna()

    df_has_text = playlist_data[has_text_cond]
    df_has_no_text = playlist_data[~has_text_cond]

    df_has_text = strip_texts(df_has_text)

    df_no_newline_in_text = df_has_text[
        df_has_text["text"].map(lambda x: r"\n" not in x)
    ]

    for idx in df_no_newline_in_text.index:
        x = df_has_text.loc[idx]
        print(f"Bad formatting: {x.name}) {x['title']} by {x['author']}")

    playlist_data = (
        pd.concat([df_has_text, df_has_no_text]).reindex().reset_index(drop=True)
    )

    playlist_data.to_csv(PATH_ALL.playlist_data, sep="|")

    copy_texts_from_df_to_yaml(df=playlist_data, path_yaml=PATH_ALL.playlist_data_yaml)


# %%

await update_songs()
# %%

# Lemmatizes non-lemmatized texts
update_lemmatized_sources(config=CONFIG)

# %%

update_word_lists(config=CONFIG)

# %%

# Requires the OPENAI_API_KEY to be loaded into the environment
await generate_deck_data_iteratively(
    config=CONFIG, generate_deck_data=generate_deck_data
)

# %%

update_dewiki_articles_dictionary()

# %%

deck_raw = read_csv(path=PATH_ALL.deck)
update_word_counts(config=CONFIG, deck_raw=deck_raw)

# %%

# Manually write responses if the generator stopped for some reason
lib.write_responses_to_deck_raw(config=CONFIG)

# %%

# Check the current prompt
print(lib.make_prompt(config=CONFIG))
