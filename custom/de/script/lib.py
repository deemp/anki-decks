from pathlib import Path
from io import StringIO
import os
import json
import math
from enum import StrEnum
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, List
import logging
import pandas as pd
import spacy
from iso639 import Lang
from pydantic import BaseModel, ValidationError
from IPython.display import display
from custom.de.script.api_request_parallel_processor import (
    process_api_requests_from_file,
)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ARTICLES_SHORT = ["f", "m", "n"]
ARTICLES_FULL = ["die", "der", "das"]
ARTICLES_DICT = dict(zip(ARTICLES_SHORT, ARTICLES_FULL))


class Model(StrEnum):
    CHATGPT_4O_MINI = "gpt-4o-mini"
    CHATGPT_4O = "chatgpt-4o"


@dataclass
class ConstRareWords:
    min_count_in_sentence: int = 2
    max_occurences_in_deck: int = 3


@dataclass
class ConstGenerationSettings:
    iterations: int = 20
    block_size: int = 70
    blocks_per_iteration: int = 20


@dataclass
class ConstSentenceLength:
    mini: int = 60
    maxi: int = 70


@dataclass(frozen=True)
class ConstPath:
    word_counts: type[Path]
    deck_raw: type[Path]
    words_bad_baseform: type[Path]
    parallel_requests: type[Path]
    parallel_responses: type[Path]
    parallel_responses_concatenated: type[Path]


@dataclass(frozen=True)
class ConstPromptSettings:
    has_part_of_speech: bool
    has_word_en: bool


@dataclass(frozen=True)
class ApiRequestsArgs:
    requests_filepath: Optional[str] = None
    save_filepath: Optional[str] = None
    request_url: Optional[str] = None
    api_key: str = OPENAI_API_KEY
    max_requests_per_minute: float = 3_000 * 0.5
    max_tokens_per_minute: float = 250_000 * 0.5
    token_encoding_name: str = "cl100k_base"
    max_attempts: int = 5
    logging_level: int = logging.INFO


@dataclass(frozen=True)
class ConstLanguageSettings:
    lang_1: type[Lang]
    "the language that you want to study"
    lang_2: type[Lang]
    "the language you know"


@dataclass(frozen=True)
class ConstColumnNames:
    word_lang_1: str
    part_of_speech_lang_1: str
    word_lang_2: str
    sentence_lang_1: str
    sentence_lang_2: str
    sentence_lemmatized_lang_1: str

    @classmethod
    def from_language_settings(cls, language_settings: type[ConstLanguageSettings]):
        lang_1 = language_settings.lang_1.pt3
        lang_2 = language_settings.lang_2.pt3
        return cls(
            word_lang_1=f"word_{lang_1}",
            part_of_speech_lang_1=f"part_of_speech_{lang_1}",
            word_lang_2=f"word_{lang_2}",
            sentence_lang_1=f"sentence_{lang_1}",
            sentence_lang_2=f"sentence_{lang_2}",
            sentence_lemmatized_lang_1=f"sentence_lemmatized_{lang_1}",
        )


@dataclass(frozen=True)
class ConstConfig:
    column_names: type[ConstColumnNames]
    rare_words: type[ConstRareWords]
    generation_settings: type[ConstGenerationSettings]
    sentence_length: type[ConstSentenceLength]
    path: type[ConstPath]
    model: type[Model]
    prompt_settings: type[ConstPromptSettings]
    nlp: type[spacy.Language]
    api_requests_args: type[ApiRequestsArgs]
    lemmatized_sep: str
    language_settings: type[ConstLanguageSettings]


def read_csv(path: type[Path]):
    return pd.read_csv(path, sep="|", index_col=0)


# TODO pass in the config
def make_baseform(word: str) -> str:
    # article
    if word in ARTICLES_FULL:
        return word
    try:
        for i, j in enumerate(word):
            if j in [".", "(", ","]:
                word = word[:i].strip()
                break
    except Exception as e:
        print(f"{word=}")
        raise e

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


def tokenize_sentence(config: type[ConstConfig], sentence: str):
    doc = config.nlp(sentence)
    tokens = [tok.lemma_ for tok in doc]

    # separable verbs
    for token in doc:
        if token.dep_ == "svp" and token.head.pos_ == "VERB":
            verb_stem = token.head.lemma_
            prefix = token.text
            tokens[token.head.i] = prefix + verb_stem

    tokens = [tok for tok in tokens if tok not in ["-", "--", " ", "  ", "„", "“"]]

    return config.lemmatized_sep.join(tokens)


def remove_separators_in_file(path: type[Path]):
    with open(path, "r", encoding="UTF-8") as d:
        deck_lines = d.readlines()

    for i, line in enumerate(deck_lines):
        for j in range(1, 10):
            if line[-j] not in ["|", "\n"]:
                deck_lines[i] = line[: -j + 1] + "\n"
                break

    with open(path, "w", encoding="UTF-8") as d:
        d.writelines(deck_lines)


def make_prompt(config: type[ConstConfig]):
    lang_1 = config.language_settings.lang_1.name
    lang_2 = config.language_settings.lang_2.name

    column_part_of_speech = f"Column 3 - lowercase part of speech of the {lang_1} word"
    column_word_en = f"Column 4 - the translation of the {lang_1} word to {lang_2}"

    add_column_part_of_speech = f"- Add {column_part_of_speech}."
    add_column_word_en = f"- Add {column_word_en}."

    has_part_of_speech = config.prompt_settings.has_part_of_speech
    has_word_en = config.prompt_settings.has_word_en

    return f"""
        Act as a native {lang_1} speaker.
    
        ## Input Table

        Column 1 - Index
        Column 2 - {lang_1} word
        {column_part_of_speech if has_part_of_speech else ""}
        {column_word_en if has_word_en else ""}

        ## Guidelines for the {lang_1} sentence:

        The sentence MUST:

        - sound natural
        - make sense
        - be engaging
        - contain the {lang_1} word (column 2)
        - be complete (have subject and verb)
        - be not too long, {config.sentence_length.mini} to {config.sentence_length.maxi} characters long
        - contain (separable) expressive verbs, concrete nouns
        - be specific, concrete
        - have no parasite words

        
        Avoid repetitive, common, simple, non-specialized words, pronouns
        Avoid philosophical and abstract thoughts.
        Prefer concrete situations and topics.
        Use sophisticated vivid thematic vocabulary.

        ## Task
        
        {add_column_part_of_speech if not has_part_of_speech else ""}
        {add_column_word_en if not has_word_en else ""}
        - Add column 5 - the sentence in {lang_1} following the sentence guidelines. The sentence must contain the {lang_1} word (column 2)
        - Add column 6 - translation of the {lang_1} sentence to {lang_2}. The translation must contain the {lang_2} word (column 4).

        Print the table as a Markdown code block CSV. 
        Use "|" as separator.
        Never print a header.
        Never skip an input row.
        Never analyze, just output.
        """


def prepare_requests(
    config: type[ConstConfig],
    deck_raw: type[pd.DataFrame],
):
    prompt = make_prompt(config=config)

    deck_raw_no_data = deck_raw.loc[
        deck_raw[config.column_names.sentence_lang_1].isna(),
        config.column_names.word_lang_1,
    ]

    if deck_raw_no_data.empty:
        raise Exception("No empty rows to generate data for!")

    blocks_per_iteration = min(
        config.generation_settings.blocks_per_iteration,
        math.ceil(deck_raw_no_data.shape[0] / config.generation_settings.block_size),
    )

    deck_raw_no_data = deck_raw_no_data.sample(frac=1).iloc[
        : config.generation_settings.block_size * blocks_per_iteration
    ]

    requests = []
    for i in range(blocks_per_iteration):
        deck_block = deck_raw_no_data.iloc[
            i
            * config.generation_settings.block_size : (i + 1)
            * config.generation_settings.block_size
        ]
        deck_str_buff = StringIO()
        deck_block.to_csv(deck_str_buff, sep="|", header=None)
        deck_str = deck_str_buff.getvalue()
        request = {
            "model": config.model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "user", "content": deck_str},
            ],
        }
        requests.append(request)

    with open(config.path.parallel_requests, "w", encoding="UTF-8") as r:
        r.writelines([json.dumps(x) + "\n" for x in requests])

    with open(config.path.parallel_responses, "w", encoding="UTF-8") as r:
        r.write("")


class ResponseContent(BaseModel, extra="allow"):
    content: str


class ResponseChoice(BaseModel, extra="allow"):
    message: type[ResponseContent]


class Response(BaseModel, extra="allow"):
    choices: List[type[ResponseChoice]]


def write_parallel_responses_concatenated(config: type[ConstConfig]):
    parallel_responses = None

    with open(config.path.parallel_responses, "r", encoding="UTF-8") as r:

        def parse_obj(x):
            try:
                return Response.parse_obj(x)
            except ValidationError as e:
                print(e)
                return None

        lines = [json.loads(x) for x in r.readlines()]
        parallel_responses = [parse_obj(x[1]) for x in lines]
        if any(not x for x in parallel_responses):
            invalid_responses = [
                line
                for line, response in zip(lines, parallel_responses)
                if not response
            ]
            print("These responses are invalid:")
            for response in invalid_responses:
                print(response)

            raise Exception(f"Invalid responses")

    parallel_responses = [
        x.choices[0]
        .message.content.strip()
        .replace("```csv\n", "")
        .replace("```\n", "")
        .replace("\n```", "")
        .strip()
        for x in parallel_responses
    ]

    parallel_responses_clean = [
        x.strip().strip("|").strip().replace(" | ", "|")
        for x in "\n".join(parallel_responses).split("\n")
    ]

    parallel_responses_clean = [
        x for x in parallel_responses_clean if x and x[0].isdecimal()
    ]

    parallel_responses_concatenated = "\n".join(parallel_responses_clean)

    with open(config.path.parallel_responses_concatenated, "w", encoding="UTF-8") as r:
        r.write(parallel_responses_concatenated)


def write_responses_to_deck_raw(config: type[ConstConfig]):
    deck_raw = read_csv(config.path.deck_raw)

    df_responses = pd.read_csv(
        config.path.parallel_responses_concatenated,
        sep="|",
        index_col=0,
        names=["index"] + list(deck_raw.columns),
    )

    df_responses = df_responses[~df_responses.index.duplicated()]

    df_responses = df_responses[
        df_responses[config.column_names.word_lang_1].notna()
        & df_responses[config.column_names.word_lang_1].map(
            lambda x: x.strip(), na_action="ignore"
        )
    ]

    in_deck_raw_cond = df_responses.index[df_responses.index.isin(deck_raw.index)]

    deck_raw.loc[in_deck_raw_cond] = df_responses.loc[in_deck_raw_cond]

    columns_strip = deck_raw.columns
    deck_raw[columns_strip] = deck_raw[columns_strip].apply(lambda x: x.str.strip())

    write_deck_raw(config=config, deck_raw=deck_raw)

    return deck_raw


def is_noun(word: str):
    return make_baseform(word)[0].isupper()


def write_deck_raw(config: type[ConstConfig], deck_raw: type[pd.DataFrame]):
    deck_raw.to_csv(config.path.deck_raw, sep="|")
    remove_separators_in_file(path=config.path.deck_raw)


def update_deck_raw_lemmatized_sentences(
    config: type[ConstConfig], deck_raw: type[pd.DataFrame]
):
    not_lemmatized_cond = deck_raw[
        config.column_names.sentence_lemmatized_lang_1
    ].isna()

    deck_raw.loc[
        not_lemmatized_cond, config.column_names.sentence_lemmatized_lang_1
    ] = deck_raw.loc[not_lemmatized_cond, config.column_names.sentence_lang_1].map(
        lambda sentence: tokenize_sentence(config=config, sentence=sentence),
        na_action="ignore",
    )

    write_deck_raw(config=config, deck_raw=deck_raw)

    return deck_raw


def update_word_counts(config: type[ConstConfig], deck_raw: type[pd.DataFrame]):
    words = pd.DataFrame(
        deck_raw[config.column_names.sentence_lemmatized_lang_1]
        .map(lambda x: x.split(";"), na_action="ignore")
        .explode(config.column_names.sentence_lemmatized_lang_1)
    ).rename(
        columns={
            config.column_names.sentence_lemmatized_lang_1: config.column_names.word_lang_1
        }
    )

    word_counts = pd.DataFrame(words.value_counts())
    word_counts.to_csv(config.path.word_counts, sep="|")

    return word_counts


def check_is_correct_sentence(
    config: type[ConstConfig],
    row: type[pd.Series],
    word_stats: type[pd.DataFrame],
    words_bad_wordforms: type[pd.DataFrame],
) -> bool:
    word_lang_1 = row[config.column_names.word_lang_1]
    word_lang_1 = (
        make_baseform(word_lang_1)
        if row.name not in words_bad_wordforms.index
        else words_bad_wordforms.loc[
            words_bad_wordforms[config.column_names.word_lang_1] == word_lang_1,
            "baseform",
        ].values[0]
    )

    sentence_lemmatized_lang_1 = row[
        config.column_names.sentence_lemmatized_lang_1
    ].split(config.lemmatized_sep)

    word_is_in_the_sentence = (
        # True
        word_lang_1
        in sentence_lemmatized_lang_1
    )

    words = pd.DataFrame(
        data=sentence_lemmatized_lang_1, columns=[config.column_names.word_lang_1]
    )

    word_counts = words.join(
        word_stats, on=config.column_names.word_lang_1, rsuffix="r"
    )

    # run this check on all sentences
    # because the number of sentences will grow
    # and the word counts will grow too
    sentence_contains_rare_words = (
        sum(
            (word_counts[config.column_names.word_lang_1] != word_lang_1)
            & (word_counts["count"] <= config.rare_words.max_occurences_in_deck)
        )
        >= config.rare_words.min_count_in_sentence
    )

    result = sentence_contains_rare_words & word_is_in_the_sentence
    # & sentence_contains_no_too_frequent_words

    # I subtract to correctly analyze other rows
    if not result:
        for word in words[config.column_names.word_lang_1]:
            word_stats.loc[word, "count"] -= 1

    return result


def partition_deck_raw_by_having_sentence_lang_1(
    config: type[ConstConfig], deck_raw: type[pd.DataFrame]
):
    has_data_cond = deck_raw[config.column_names.sentence_lang_1].notna()
    rows_with_data = deck_raw[has_data_cond].sort_index()
    rows_without_data = deck_raw[~has_data_cond].sort_index()

    deck_raw = pd.concat([rows_with_data, rows_without_data])
    write_deck_raw(config=config, deck_raw=deck_raw)

    return rows_with_data, rows_without_data


def check_sentence_length(config: type[ConstConfig], row: pd.Series()):
    return (
        pd.notna(row[config.column_names.sentence_lang_1])
        and config.sentence_length.mini
        <= len(row[config.column_names.sentence_lang_1])
        <= config.sentence_length.maxi
    )


def filter_deck_raw_by_sentence_length(
    config: type[ConstConfig],
    deck_raw: type[pd.DataFrame],
):
    rows_with_data, rows_without_data = partition_deck_raw_by_having_sentence_lang_1(
        config=config, deck_raw=deck_raw
    )

    is_good_sentence_length_cond = rows_with_data.apply(
        lambda row: check_sentence_length(config=config, row=row),
        axis=1,
    )
    rows_with_good_sentence_length = rows_with_data[is_good_sentence_length_cond]
    rows_with_bad_sentence_length = pd.DataFrame(
        rows_with_data.loc[
            ~is_good_sentence_length_cond,
            [
                config.column_names.word_lang_1,
                config.column_names.part_of_speech_lang_1,
                config.column_names.word_lang_2,
            ],
        ]
    )

    rows_without_data = pd.concat(
        [rows_with_bad_sentence_length, rows_without_data]
    ).sort_index()[config.column_names.word_lang_1]

    deck_raw = pd.concat([rows_with_good_sentence_length, rows_without_data])

    write_deck_raw(config=config, deck_raw=deck_raw)

    return deck_raw


def update_words_bad_baseform(config: type[ConstConfig], deck_raw: type[pd.DataFrame]):
    words_bad_baseform = read_csv(config.path.words_bad_baseform)

    words_bad_baseform = words_bad_baseform.join(
        deck_raw[config.column_names.word_lang_1]
        .reset_index()
        .set_index(config.column_names.word_lang_1),
        on=config.column_names.word_lang_1,
        rsuffix="_r",
    )

    # select words that are in the deck
    words_bad_baseform = words_bad_baseform[words_bad_baseform["index"].notna()]

    words_bad_baseform.index = words_bad_baseform["index"]

    words_bad_baseform.sort_index(inplace=True)

    words_bad_baseform.drop(columns=["index"], inplace=True)

    words_bad_baseform.index.name = None

    words_bad_baseform.to_csv(config.path.words_bad_baseform, sep="|")

    return words_bad_baseform


def partition_deck_raw(
    config: type[ConstConfig],
    deck_raw: type[pd.DataFrame],
):
    deck_raw = filter_deck_raw_by_sentence_length(config=config, deck_raw=deck_raw)

    deck_raw = update_deck_raw_lemmatized_sentences(config=config, deck_raw=deck_raw)

    word_stats = update_word_counts(config=config, deck_raw=deck_raw)

    rows_with_data, rows_without_data = partition_deck_raw_by_having_sentence_lang_1(
        config=config, deck_raw=deck_raw
    )

    words_bad_baseform = update_words_bad_baseform(config=config, deck_raw=deck_raw)

    # prefer removing rows with a larger index
    # to change rows with smaller index less frequently
    # to preserve progress
    #
    # rows with smaller indices may get removed
    # if we leave rows with larger indices that introduce some words
    # that increase word counts
    # and hence disallow rows with smaller indices to stay
    is_correct_sentence_cond = (
        rows_with_data.iloc[::-1]
        .apply(
            lambda row: check_is_correct_sentence(
                config=config,
                row=row,
                word_stats=word_stats,
                words_bad_wordforms=words_bad_baseform,
            ),
            axis=1,
        )
        .sort_index()
    )

    has_correct_sentence = rows_with_data[is_correct_sentence_cond]
    has_incorrect_sentence = rows_with_data[~is_correct_sentence_cond]
    has_incorrect_sentence = has_incorrect_sentence[
        [
            config.column_names.word_lang_1,
            config.column_names.part_of_speech_lang_1,
            config.column_names.word_lang_2,
        ]
    ]

    has_no_data = pd.concat([has_incorrect_sentence, rows_without_data]).sort_index()

    deck_raw = pd.concat([has_correct_sentence, has_no_data])
    deck_raw = deck_raw[~deck_raw.index.duplicated()]

    write_deck_raw(config=config, deck_raw=deck_raw)

    return deck_raw


async def update_deck_raw(config: type[ConstConfig]):
    def partition(deck_raw: type[pd.DataFrame]):
        return partition_deck_raw(config=config, deck_raw=deck_raw)

    deck_raw = read_csv(config.path.deck_raw)

    # deck_raw = partition(deck_raw=deck_raw)

    prepare_requests(
        config=config,
        deck_raw=deck_raw,
    )

    await process_api_requests_from_file(
        requests_filepath=config.api_requests_args.requests_filepath,
        save_filepath=config.api_requests_args.save_filepath,
        request_url=config.api_requests_args.request_url,
        api_key=config.api_requests_args.api_key,
        max_requests_per_minute=config.api_requests_args.max_requests_per_minute,
        max_tokens_per_minute=config.api_requests_args.max_tokens_per_minute,
        token_encoding_name=config.api_requests_args.token_encoding_name,
        max_attempts=config.api_requests_args.max_attempts,
        logging_level=config.api_requests_args.logging_level,
    )

    write_parallel_responses_concatenated(config=config)

    deck_raw = write_responses_to_deck_raw(config=config)

    deck_raw = partition(deck_raw=deck_raw)

    return deck_raw


async def generate_deck_data_iteratively(
    config: type[ConstConfig],
    generate_deck_data: Callable[[type[ConstConfig]], Awaitable[None]],
):
    for i in range(config.generation_settings.iterations):
        print(f"Iteration: {i}")

        await generate_deck_data(config)
