from pathlib import Path
from io import StringIO
import os
import json
import math
import pandas as pd
import spacy
from enum import StrEnum
from custom.de.script.api_request_parallel_processor import (
    process_api_requests_from_file,
)
from dataclasses import dataclass
from typing import Optional
import logging

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ARTICLES_SHORT = ["f", "m", "n"]
ARTICLES_FULL = ["die", "der", "das"]
ARTICLES_DICT = dict(zip(ARTICLES_SHORT, ARTICLES_FULL))
DEWIKI_ARTICLES_SEP = ";"
LEMMATIZED_SEP = ";"

data_path = Path(f'{os.environ["ROOT_DIR"]}/custom/de/data')

DEWIKI_NOUN_ARTICLES = pd.read_csv(data_path / "dewiki-noun-articles.csv", sep="|")
LEMMATA = pd.read_csv("data/dwds_lemmata_2025-01-15.csv")


class MODEL(StrEnum):
    CHATGPT_4O_MINI = "gpt-4o-mini"
    CHATGPT_4O = "chatgpt-4o"


def read_csv(path: type[Path]):
    return pd.read_csv(path, sep="|", index_col=0)


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
        raise

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


def tokenize_sentence(sentence: str, nlp: type[spacy.Language]):
    doc = nlp(sentence)
    tokens = [tok.lemma_ for tok in doc]

    # separable verbs
    for token in doc:
        if token.dep_ == "svp" and token.head.pos_ == "VERB":
            verb_stem = token.head.lemma_
            prefix = token.text
            tokens[token.head.i] = prefix + verb_stem

    tokens = [tok for tok in tokens if tok not in ["-", "--", " ", "  ", "„", "“"]]

    return LEMMATIZED_SEP.join(tokens)


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


def make_is_lemma_cond(s: type[pd.Series]):
    s = s.map(lambda x: make_baseform(x) if isinstance(x, str) else False)
    return s.isin(LEMMATA["lemma"]) | s.isin(DEWIKI_NOUN_ARTICLES["lemma"])


def make_prompt(has_part_of_speech: bool, has_word_en: bool):
    column_part_of_speech = "Column 3 - part of speech"
    column_word_en = "Column 4 - English word"

    add_column_part_of_speech = (
        "- Add column 3 - lowercase part of speech of the German word."
    )
    add_column_word_en = (
        "- Add column 4 - the translation of the German word to English."
    )

    return f"""
        Act as a true German.
    
        ## Input Table

        Column 1 - index
        Column 2 - German word
        {column_part_of_speech if has_part_of_speech else ""}
        {column_word_en if has_word_en else ""}

        ## Guidelines for the German sentence:

        The sentence MUST:

        - sound natural
        - make sense
        - be engaging
        - contain the German word (column 2)
        - be complete (have subject and verb)
        - be not too long, 60 to 65 characters long
        - contain separable expressive verbs, concrete nouns
        - be specific, concrete
        - have no parasite words like "besonders"

        ## Avoid

        repetitive, common, simple, non-specialized words, pronouns

        ## Task
        
        {add_column_part_of_speech if not has_part_of_speech else ""}
        {add_column_word_en if not has_word_en else ""}
        - Add column 5 - the sentence in German following the sentence guidelines. The sentence must contain the German word (column 2)
        - Add column 6 - translation of the German sentence to English. The translation must contain the English word (column 4).

        Print the table as markdown code block CSV with "|" as separator.
        Never print a header.
        Never skip an input row.
        Never analyze, just output.
        
        Avoid philosophical and abstract thoughts. 
        Prefer concrete situations and topics.
        Use sophisticated vivid thematic vocabulary.

        I'll provide input in the next messages.
        """


def prepare_requests(
    deck_raw: type[pd.DataFrame],
    part_count_per_iteration: int,
    part_size: int,
    model: MODEL,
    parallel_requests_path: type[Path],
    parallel_responses_path: type[Path],
    has_part_of_speech: bool,
    has_word_en: bool,
):
    prompt = make_prompt(has_part_of_speech=has_part_of_speech, has_word_en=has_word_en)

    deck_raw_no_data = deck_raw[deck_raw["sentence_de"].isna()]

    part_count_per_iteration = min(
        part_count_per_iteration, math.ceil(deck_raw_no_data.shape[0] / part_size)
    )

    deck_raw_no_data = deck_raw_no_data.sample(frac=1).iloc[
        : part_size * part_count_per_iteration
    ]

    requests = []
    for i in range(part_count_per_iteration):
        deck_block = deck_raw_no_data.iloc[i * part_size : (i + 1) * part_size]
        deck_str_buff = StringIO()
        deck_block.to_csv(deck_str_buff, sep="|", header=None)
        deck_str = deck_str_buff.getvalue().replace("|||", "")
        request = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "user", "content": deck_str},
            ],
        }
        requests.append(request)

    with open(parallel_requests_path, "w", encoding="UTF-8") as r:
        r.writelines([json.dumps(x) + "\n" for x in requests])

    with open(parallel_responses_path, "w", encoding="UTF-8") as r:
        r.write("")


def write_parallel_responses(
    parallel_responses_path: type[Path],
    parallel_responses_concatenated_path: type[Path],
):
    parallel_responses = None

    with open(parallel_responses_path, "r", encoding="UTF-8") as r:
        parallel_responses = [json.loads(x) for x in r.readlines()]

    parallel_responses = [
        x[1]["choices"][0]["message"]["content"]
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

    with open(parallel_responses_concatenated_path, "w", encoding="UTF-8") as r:
        r.write(parallel_responses_concatenated)


def write_responses_to_deck_raw(
    parallel_responses_concatenated_path: type[Path], deck_raw_path: type[Path]
):
    df_responses = pd.read_csv(
        parallel_responses_concatenated_path, sep="|", index_col=0, header=None
    )

    deck_raw = pd.read_csv(deck_raw_path, sep="|", index_col=0)

    df_responses = df_responses[~df_responses.index.duplicated()]
    in_deck_raw_cond = df_responses.index[df_responses.index.isin(deck_raw.index)]
    df_responses["sentence_lemmatized_de"] = None
    df_responses.columns = deck_raw.columns


    deck_raw.loc[in_deck_raw_cond] = df_responses.loc[in_deck_raw_cond]

    columns_strip = ["word_de", "part_of_speech", "word_en", "sentence_de", "sentence_en"]
    deck_raw[columns_strip] = deck_raw[columns_strip].apply(lambda x: x.str.strip())

    deck_raw.to_csv(deck_raw_path, sep="|")

    return deck_raw


def is_noun(word: str):
    return make_baseform(word)[0].isupper()


def write_deck_raw(deck_raw_path: type[Path], deck_raw: type[pd.DataFrame]):
    deck_raw.to_csv(deck_raw_path, sep="|")
    remove_separators_in_file(path=deck_raw_path)


def update_deck_raw_lemmatized_sentences(
    deck_raw: type[pd.DataFrame], deck_raw_path: type[Path], nlp: type[spacy.Language]
):
    not_lemmatized_cond = deck_raw["sentence_lemmatized_de"].isna()

    deck_raw.loc[not_lemmatized_cond, "sentence_lemmatized_de"] = deck_raw.loc[
        not_lemmatized_cond, "sentence_de"
    ].map(lambda x: tokenize_sentence(sentence=x, nlp=nlp), na_action="ignore")

    write_deck_raw(deck_raw=deck_raw, deck_raw_path=deck_raw_path)

    return deck_raw


def update_word_counts(deck_raw: type[pd.DataFrame], word_counts_path: type[Path]):
    words = pd.DataFrame(
        deck_raw["sentence_lemmatized_de"]
        .map(lambda x: x.split(";"), na_action="ignore")
        .explode("sentence_lemmatized_de")
    ).rename(columns={"sentence_lemmatized_de": "word_de"})

    word_counts = pd.DataFrame(words.value_counts())
    word_counts.to_csv(word_counts_path, sep="|")

    return word_counts


def check_is_correct_sentence(
    row: type[pd.Series],
    word_stats: type[pd.DataFrame],
    words_bad_wordforms: type[pd.DataFrame],
    max_occurences: int,
) -> bool:
    word_de = row["word_de"]
    word_de = (
        make_baseform(word_de)
        if row.name not in words_bad_wordforms.index
        else words_bad_wordforms.loc[
            words_bad_wordforms["word_de"] == word_de, "baseform"
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
    sentence_contains_rare_words = (
        sum(
            (word_counts["word_de"] != word_de)
            & (word_counts["count"] <= max_occurences)
        )
        >= 3
    )

    result = sentence_contains_rare_words & word_is_in_the_sentence
    # & sentence_contains_no_too_frequent_words

    # I subtract to correctly analyze other rows
    if not result:
        for word in words["word_de"]:
            word_stats.loc[word, "count"] -= 1

    return result


def partition_deck_raw_by_having_sentence_de(
    deck_raw: type[pd.DataFrame], deck_raw_path: type[Path]
):
    has_data_cond = deck_raw["sentence_de"].notna()
    rows_with_data = deck_raw[has_data_cond].sort_index()
    rows_without_data = deck_raw[~has_data_cond].sort_index()

    deck_raw = pd.concat([rows_with_data, rows_without_data])
    write_deck_raw(deck_raw=deck_raw, deck_raw_path=deck_raw_path)

    return rows_with_data, rows_without_data


def check_sentence_length(
    x: pd.Series(), sentence_length_min: int, sentence_length_max: int
):
    return (
        pd.notna(x["sentence_de"])
        and sentence_length_min <= len(x["sentence_de"]) <= sentence_length_max
    )


def filter_deck_raw_by_sentence_length(
    deck_raw: type[pd.DataFrame],
    deck_raw_path: type[Path],
    sentence_length_min: int,
    sentence_length_max: int,
):
    rows_with_data, rows_without_data = partition_deck_raw_by_having_sentence_de(
        deck_raw=deck_raw, deck_raw_path=deck_raw_path
    )

    is_good_sentence_length_cond = rows_with_data.apply(
        lambda x: check_sentence_length(
            x=x,
            sentence_length_min=sentence_length_min,
            sentence_length_max=sentence_length_max,
        ),
        axis=1,
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

    deck_raw = pd.concat([rows_with_good_sentence_length, rows_without_data])

    write_deck_raw(deck_raw=deck_raw, deck_raw_path=deck_raw_path)

    return deck_raw


def update_words_bad_baseform(
    deck_raw: type[pd.DataFrame], words_bad_baseform_path: type[Path]
):
    words_bad_baseform = read_csv(words_bad_baseform_path)

    words_bad_baseform = words_bad_baseform.join(
        deck_raw["word_de"].reset_index().set_index("word_de"),
        on="word_de",
        rsuffix="_r",
    )

    # select words that are in the deck
    words_bad_baseform = words_bad_baseform[words_bad_baseform["index"].notna()]

    words_bad_baseform.index = words_bad_baseform["index"]

    words_bad_baseform.sort_index(inplace=True)

    words_bad_baseform.drop(columns=["index"], inplace=True)

    words_bad_baseform.index.name = None

    words_bad_baseform.to_csv(words_bad_baseform_path, sep="|")

    return words_bad_baseform


def partition_deck_raw(
    deck_raw: type[pd.DataFrame],
    word_counts_path: type[Path],
    max_occurences: int,
    words_bad_baseform_path: type[Path],
    deck_raw_path: type[Path],
    nlp: type[spacy.Language],
    sentence_length_min: int,
    sentence_length_max: int,
):
    deck_raw = filter_deck_raw_by_sentence_length(
        deck_raw=deck_raw,
        deck_raw_path=deck_raw_path,
        sentence_length_min=sentence_length_min,
        sentence_length_max=sentence_length_max,
    )

    deck_raw = update_deck_raw_lemmatized_sentences(
        deck_raw=deck_raw, deck_raw_path=deck_raw_path, nlp=nlp
    )

    word_stats = update_word_counts(
        deck_raw=deck_raw, word_counts_path=word_counts_path
    )

    rows_with_data, rows_without_data = partition_deck_raw_by_having_sentence_de(
        deck_raw=deck_raw, deck_raw_path=deck_raw_path
    )

    words_bad_baseform = update_words_bad_baseform(
        deck_raw=deck_raw, words_bad_baseform_path=words_bad_baseform_path
    )

    # TODO

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
                row=row,
                word_stats=word_stats,
                words_bad_wordforms=words_bad_baseform,
                max_occurences=max_occurences,
            ),
            axis=1,
        )
        .sort_index()
    )

    has_correct_sentence = rows_with_data[is_correct_sentence_cond]
    has_incorrect_sentence = rows_with_data[~is_correct_sentence_cond]
    has_incorrect_sentence = has_incorrect_sentence[
        ["word_de", "part_of_speech", "word_en"]
    ]

    rows_without_data = pd.concat(
        [has_incorrect_sentence, rows_without_data]
    ).sort_index()

    deck_raw = pd.concat([has_correct_sentence, rows_without_data])
    deck_raw = deck_raw[~deck_raw.index.duplicated()]

    write_deck_raw(deck_raw=deck_raw, deck_raw_path=deck_raw_path)

    return deck_raw


@dataclass
class APIRequestsArgs:
    requests_filepath: Optional[str] = None
    save_filepath: Optional[str] = None
    request_url: Optional[str] = None
    api_key: str = OPENAI_API_KEY
    max_requests_per_minute: float = 3_000 * 0.5
    max_tokens_per_minute: float = 250_000 * 0.5
    token_encoding_name: str = "cl100k_base"
    max_attempts: int = 5
    logging_level: int = logging.INFO


async def update_deck_raw(
    word_counts_path: type[Path],
    deck_raw_path: type[Path],
    max_occurences: int,
    words_bad_baseform_path: type[Path],
    part_count_per_iteration: int,
    part_size: int,
    parallel_requests_path: type[Path],
    parallel_responses_path: type[Path],
    parallel_responses_concatenated_path: type[Path],
    api_requests_args: type[APIRequestsArgs],
    model: MODEL,
    nlp: type[spacy.Language],
    sentence_length_min: int,
    sentence_length_max: int,
    has_part_of_speech: bool,
    has_word_en: bool,
):
    deck_raw = pd.read_csv(deck_raw_path, sep="|", index_col=0)

    prepare_requests(
        deck_raw=deck_raw,
        part_count_per_iteration=part_count_per_iteration,
        part_size=part_size,
        model=model,
        parallel_requests_path=parallel_requests_path,
        parallel_responses_path=parallel_responses_path,
        has_part_of_speech=has_part_of_speech,
        has_word_en=has_word_en,
    )

    await process_api_requests_from_file(
        requests_filepath=api_requests_args.requests_filepath,
        save_filepath=api_requests_args.save_filepath,
        request_url=api_requests_args.request_url,
        api_key=api_requests_args.api_key,
        max_requests_per_minute=api_requests_args.max_requests_per_minute,
        max_tokens_per_minute=api_requests_args.max_tokens_per_minute,
        token_encoding_name=api_requests_args.token_encoding_name,
        max_attempts=api_requests_args.max_attempts,
        logging_level=api_requests_args.logging_level,
    )

    write_parallel_responses(
        parallel_responses_path=parallel_responses_path,
        parallel_responses_concatenated_path=parallel_responses_concatenated_path,
    )

    deck_raw = write_responses_to_deck_raw(
        parallel_responses_concatenated_path=parallel_responses_concatenated_path,
        deck_raw_path=deck_raw_path,
    )

    partition_deck_raw(
        deck_raw=deck_raw,
        word_counts_path=word_counts_path,
        deck_raw_path=deck_raw_path,
        max_occurences=max_occurences,
        words_bad_baseform_path=words_bad_baseform_path,
        nlp=nlp,
        sentence_length_min=sentence_length_min,
        sentence_length_max=sentence_length_max,
    )
