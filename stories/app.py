#%%
import threading
from bs4 import BeautifulSoup
import time
import concurrent.futures
import requests
from nltk import tokenize
import nltk
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import quote
from time import sleep
import shutil
from textwrap import dedent
from itertools import takewhile
from dataclasses import dataclass
from selenium.webdriver.firefox.options import Options

config = [
    {"url": "https://www.eslfast.com/kidsenglish/ke/ke", "count": 100},
    {"url": "https://www.eslfast.com/kidsenglish2/ke2/ke2", "count": 100},
    {"url": "https://www.eslfast.com/kidsenglish3/ke3/ke3", "count": 100},
    {"url": "https://www.eslfast.com/supereasy/se/supereasy", "count": 200},
    {"url": "https://www.eslfast.com/easyread/es/easy", "count": 200},
    {"url": "https://www.eslfast.com/begin1/b1/b1", "count": 104},
    {"url": "https://www.eslfast.com/begin2/b2/b2", "count": 110},
    {"url": "https://www.eslfast.com/begin3/b3/b3", "count": 110},
    {"url": "https://www.eslfast.com/begin4/b4/b4", "count": 100},
    {"url": "https://www.eslfast.com/begin5/b5/b5", "count": 100},
    {"url": "https://www.eslfast.com/begin6/b6/b6", "count": 100},
    {"url": "https://www.eslfast.com/gradedread1/gr/gr1", "count": 100},
]

data_dir = Path("data")

# csv
sentences_dir = "sentences"
headers_dir = "headers"
decks_dir = "decks"
texts_dir = "texts"
progress_dir = "progress"

mk_csv = lambda subdir, name: data_dir / subdir / f"{name}.csv"
mk_sentences = lambda lang: mk_csv(sentences_dir, lang)
mk_headers = lambda lang: mk_csv(headers_dir, lang)
mk_deck = lambda in_lang, out_lang: mk_csv(decks_dir, f"{in_lang}-{out_lang}")
mk_progress = lambda name: mk_csv(progress_dir, name)
# xml
mk_texts = lambda lang: data_dir / "texts" / f"{lang}.xml"

# deepl_translated = dir / "deepl_translated.csv"


def prepare(x):
    return x.strip().replace("\n", " ")


def format_text(index, header, body):
    return (
        dedent(
            f"""\
        <txt>
        <n>{index}</n>
        <header>{prepare(header)}</header>
        <body>\n"""
        )
        + "\n".join(body)
        + "\n"
        + dedent(
            """\
        </body>
        </txt>\n"""
        )
    )


def write_texts_en(texts_path, offset=0):
    urls = []
    for c in config:
        base_url = c["url"]
        count = c["count"]
        urls += [f"{base_url}{i:03}.htm" for i in range(1, count + 1)]

    # urls = urls[:10]

    def launch():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.map(requests.get, urls)

    responses = launch()

    with open(texts_path, "a", encoding="utf8") as texts_file:
        for index, response in enumerate(responses, 1):
            page = response.text
            soup = BeautifulSoup(page, "html.parser")

            def dbg():
                print(urls[index], "\n")
                print(header, "\n")
                print(body, "\n")
                print(page, "\n")

            header = None
            try:
                header = soup.find("h1", class_="main-header")
            except:
                dbg()
            # if header is None:
            #     dbg()
            body = None
            try:
                body = soup.find("div", class_="content-body").find("p").text
            except:
                dbg()
            # if body is None:
            # try:
            #     prepare(body)
            # except:
            #     print(urls[index],"\n")
            #     print(header,"\n")
            #     print(body,"\n")
            #     print(page,"\n")
            # print(soup.find("div", ))
            # print(prepare(body))
            xs = [x.strip() for x in tokenize.sent_tokenize(prepare(body))]
            texts_file.write(f"{format_text(index + offset, header.text, xs)}\n")


def write_data_en(sentences_path, headers_path, texts_path):
    with open(texts_path, "r", encoding="utf8") as tx:
        # write sentences
        soup = BeautifulSoup(tx.read(), "html.parser")
        sentences = []
        for body_tag in soup.find_all("body"):
            body = body_tag.text.strip().split("\n")
            for line in body:
                suff_dot = "." if line[-1] == '"' else ""
                y_fixed = line.replace("/", ".")
                sentences += [f"{y_fixed}{suff_dot}"]
        with open(sentences_path, "w", encoding="utf8") as sentences_file:
            sentences_file.write("\n".join(sentences) + "\n")

        # write headers
        headers = "\n".join(
            [x.text.replace("\n", " ").strip() for x in soup.find_all("header")]
        )
        with open(headers_path, "w", encoding="utf8") as headers_file:
            headers_file.write(headers + "\n")


def write_texts(to_sentences_path, to_headers_path, from_texts_path, to_texts_path):
    with open(from_texts_path, "r", encoding="utf8") as texts_file:
        soup = BeautifulSoup(texts_file.read(), "html.parser")
        with open(to_sentences_path, "r", encoding="utf8") as in_sentences_file:
            in_sentences = [x.strip() for x in in_sentences_file.readlines()]
            with open(to_headers_path, "r", encoding="utf8") as in_headers_file:
                in_headers = [x.strip() for x in in_headers_file.readlines()]
                with open(to_texts_path, "w", encoding="utf8") as out_texts_file:
                    in_bodies = [x.text.strip() for x in soup.find_all("body")]
                    total_lines = 0
                    for index, (body, header) in enumerate(
                        zip(in_bodies, in_headers), 1
                    ):
                        start_line = total_lines
                        end_line = start_line + len(body.strip().split("\n"))
                        total_lines = end_line
                        out_texts_file.write(
                            format_text(
                                index,
                                header,
                                in_sentences[start_line:end_line],
                            )
                        )


def count_block_symbols(block):
    # lengths of strings + number of newlines
    return sum(map(len, block)) + len(block)


def make_blocks_with_margins(lines, margin, max_symbols):
    """
    split sentences into blocks
    each block has a margin on both ends
    this margin is several sentences long just for context
    """

    lines_adjusted = [""] * margin + [x.strip() for x in lines] + [""] * margin
    line_lengths = [len(x) + 1 for x in lines_adjusted]

    def get_blocks():
        # (start index, end index inclusive)
        block_borders = []
        block_symbol_number = 0
        block_start_index = 0
        after_last_line_index = len(lines_adjusted) - margin + 1
        for k in range(margin, after_last_line_index):
            j = k + 1
            if block_symbol_number + sum(line_lengths[k : j + margin + 1]) > max_symbols:
                block_borders += [(block_start_index, k + margin)]
                block_start_index = j - margin
                block_symbol_number = sum(line_lengths[j - margin : j])
            else:
                block_symbol_number += line_lengths[k]
            if j == after_last_line_index:
                block_borders += [(block_start_index, k + margin)]

        return [lines_adjusted[x[0] : x[1] + 1] for x in block_borders]

    return get_blocks()


def mkrange(start, end, maxi=10**10, start_offset=0):
    return f"[ {start_offset + start} ; {min(start_offset + end, maxi)} )"


def translate(
    from_language,
    to_language,
    from_lines_path,
    to_lines_path,
    start_line_index=1,
    end_line_index=25000,
    max_symbols=3000,
    wait_query=20,
    margin=3,
    need_translate=False,
    need_copy_translation=True,
    max_workers=2,
    use_headless=True,
):

    end_line_index_adjusted = end_line_index
    from_lines = None
    with open(from_lines_path, "r", encoding="utf8") as from_lines_file:
        from_lines = from_lines_file.readlines()
        end_line_index_adjusted = min(len(from_lines), end_line_index_adjusted)
        from_lines = from_lines[start_line_index - 1 : end_line_index_adjusted]
    to_progress_path = mk_progress(
        f"{from_language}-{to_language}-{start_line_index}-{end_line_index_adjusted}"
    )
    if need_translate:
        print(f"Translating from `{from_language}` to `{to_language}`")
        base_addr = f"https://www.deepl.com/translator#{from_language}/{to_language}/"

        lock = threading.Lock()

        @dataclass
        class Task:
            browser: any
            block_index: int
            block_lines: any

        def write_translation(tasks):
            try:
                for task in tasks:
                    browser = task.browser
                    block_index = task.block_index
                    block_lines = task.block_lines

                    browser.delete_all_cookies()
                    block = "\n".join(block_lines)
                    try:
                        browser.get(f"{base_addr}{quote(block)}")
                        field = WebDriverWait(browser, 12).until(
                            EC.presence_of_element_located(
                                (
                                    By.XPATH,
                                    "/html/body/div[3]/main/div[5]/div[1]/div[2]/section[2]/div[3]/div[1]/d-textarea/div",
                                )
                            )
                        )

                        sleep(wait_query)

                        def pad(lines):
                            pref_len = len(
                                list(takewhile(lambda x: not x, block_lines))
                            )
                            suff_len = len(
                                list(takewhile(lambda x: not x, block_lines[::-1]))
                            )
                            return [""] * pref_len + lines + [""] * suff_len

                        translated_lines = pad(field.text.split("\n"))
                        block_lines_number = len(block_lines) - 2 * margin
                        translated_lines_number = len(translated_lines) - 2 * margin
                        if block_lines_number != translated_lines_number:
                            raise Exception(
                                f"Length mismatch for block: {block_index}. Expected: {block_lines_number}. Got: {translated_lines_number}"
                            )
                        else:
                            print(
                                f"Block: {block_index}. Original/Translation lengths (lines): {block_lines_number}/{translated_lines_number}"
                            )

                        with lock:
                            with open(
                                to_progress_path, "a+", encoding="utf8"
                            ) as progress_file:
                                progress_file.write(
                                    dedent(
                                        f"""\
                                        <block>
                                        <index>{block_index:04}</index>
                                        <body>
                                        """
                                    )
                                    + "\n".join(translated_lines[margin:-margin])
                                    + dedent(
                                        """
                                        </body>
                                        </block>
                                        """
                                    )
                                )
                    except Exception as e:
                        print(e)
                        raise (Exception("Stopping"))
            finally:
                browser.close()
                browser.quit()

        options = Options()
        if use_headless:
            options.add_argument("-headless")

        blocks = make_blocks_with_margins(
            lines=from_lines, margin=margin, max_symbols=max_symbols
        )

        print(
            f"Need to process lines: {mkrange(start=start_line_index, end=end_line_index_adjusted + 1)}"
        )

        with open(to_progress_path, "a+", encoding="utf8"):
            pass

        done_block_indices = {}
        with open(to_progress_path, "r", encoding="utf8") as progress_file:
            done_block_indices = {
                int(x.text)
                for x in BeautifulSoup(progress_file.read(), "html.parser").find_all(
                    "index"
                )
            }

        def plan_tasks():
            block_start_line_index = 0
            todo_blocks = list(
                filter(lambda i: i[0] not in done_block_indices, enumerate(blocks))
            )
            for block_index, block_lines in todo_blocks:
                block_end_line_index = (
                    block_start_line_index + len(block_lines) - 2 * margin
                )
                print(
                    f"Block [ {block_index + 1} / {len(blocks)} ]. "
                    + f"Lines {mkrange(block_start_line_index,  block_end_line_index, start_offset=start_line_index, maxi = end_line_index_adjusted + 1)}"
                )
                block_start_line_index = block_end_line_index
            return todo_blocks

        todo_blocks = plan_tasks()

        if len(todo_blocks) == 0:
            print("Completed!")
        else:
            try:
                max_browsers = min(max_workers, len(todo_blocks))
                browsers = [
                    webdriver.Firefox(
                        options=options, executable_path="/usr/local/bin/geckodriver"
                    )
                    for _ in range(max_browsers)
                ]

                def get_tasks():
                    tasks = [[] for _ in range(max_browsers)]
                    for block_index, block_lines in todo_blocks:
                        browser_index = block_index % max_browsers
                        tasks[browser_index] += [
                            Task(
                                browser=browsers[browser_index],
                                block_index=block_index,
                                block_lines=block_lines,
                            )
                        ]
                    return tasks

                tasks = get_tasks()

                def run_tasks():
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_browsers
                    ) as executor:
                        executor.map(
                            lambda browser_tasks: write_translation(
                                tasks=browser_tasks
                            ),
                            tasks,
                        )

                run_tasks()

                print("Completed!")
            finally:
                # pass
                for browser in browsers:
                    browser.close()
                    browser.quit()

    if need_copy_translation:

        def get_lines():
            lines = None
            with open(to_progress_path, "r", encoding="utf8") as progress_file:
                lines = progress_file.read()

            soup = BeautifulSoup(lines, "html.parser")
            blocks = soup.find_all("block")
            block_map = {}
            for block in blocks:
                index = block.find("index")
                body = block.find("body")
                block_map.update({index.text.strip(): body.text.strip()})

            block_lines = [
                x[1] for x in sorted(block_map.items(), key=lambda item: int(item[0]))
            ]
            return block_lines

        with open(to_lines_path, "w", encoding="utf8") as to_lines_file:
            to_lines_file.write("\n".join(get_lines()))


def write_deck(
    from_texts_path: str, to_sentences_path: str, to_headers_path: str, deck_path: str
):
    """
    given the texts in a `from` language, given the sentences and headers in a `to` language, write a deck
    """
    with open(from_texts_path, "r", encoding="utf8") as from_texts_file:
        soup = BeautifulSoup(from_texts_file.read(), "html.parser")
        with open(to_sentences_path, "r", encoding="utf8") as to_sentences_file:
            to_sentences = [x.strip() for x in to_sentences_file.readlines()]
            with open(to_headers_path, "r", encoding="utf8") as to_headers_file:
                to_headers = [x.strip() for x in to_headers_file.readlines()]
                with open(deck_path, "w", encoding="utf8") as deck_file:
                    cards = []
                    total_lines = 0
                    for text_index, (from_body, from_header, to_header) in enumerate(
                        zip(soup.find_all("body"), soup.find_all("header"), to_headers),
                        start=1,
                    ):
                        from_body = from_body.text.strip().split("\n")
                        start_line = total_lines
                        end_line = start_line + len(from_body)
                        total_lines = end_line
                        from_header = from_header.text.strip().replace("\n", " ")
                        to_body = to_sentences[start_line:end_line]
                        for sentence_index, (from_sentence, to_sentence) in enumerate(
                            zip(from_body, to_body)
                        ):
                            from_prev = ""
                            from_next = ""
                            to_prev = ""
                            to_next = ""
                            if sentence_index != 0:
                                from_prev = from_body[sentence_index - 1]
                                to_prev = to_body[sentence_index - 1]
                            if sentence_index != len(from_body) - 1:
                                from_next = from_body[sentence_index + 1]
                                to_next = to_body[sentence_index + 1]
                            sentence_number = start_line + sentence_index + 1
                            from_tokens_number = len(from_sentence.split(" "))
                            to_tokens_number = len(to_sentence.split(" "))                            
                            sep = "|"
                            cards += (
                                sep.join(
                                    [
                                        f"{sentence_number:05}",
                                        f"{text_index:04}",
                                        from_sentence,
                                        to_sentence,
                                        from_prev,
                                        to_prev,
                                        from_next,
                                        to_next,
                                        from_header,
                                        to_header,
                                        f"{from_tokens_number:02}",
                                        f"{to_tokens_number:02}",
                                    ]
                                ).replace('"', '""')
                                + "\n"
                            )
                    deck_file.writelines(cards)

def full_translate(
    from_language,
    to_language,
    from_sentences_path,
    from_headers_path,
    from_texts_path,
    sentence_start_line_index,
    header_start_line_index,
    sentence_end_line_index,
    header_end_line_index,
    need_translate_sentences=False,
    need_translate_headers=False,
    need_write_texts=False,
    need_translate=False,
    need_copy_translation=True,
    max_workers=10,
    wait_query=60,
    use_headless=True,
):
    print(f"Language pair: {from_language} - {to_language}")
    to_sentences_path = mk_sentences(to_language)
    to_headers_path = mk_headers(to_language)

    if need_translate_sentences:
        print(f"Translating sentences")
        # translate sentences
        translate(
            from_language=from_language,
            to_language=to_language,
            from_lines_path=from_sentences_path,
            to_lines_path=to_sentences_path,
            start_line_index=sentence_start_line_index,
            end_line_index=sentence_end_line_index,
            need_translate=need_translate,
            need_copy_translation=need_copy_translation,
            max_workers=max_workers,
            use_headless=use_headless,
            wait_query=wait_query,
        )

    if need_translate_headers:
        print(f"Translating headers")
        # translate headers
        translate(
            from_language=from_language,
            to_language=to_language,
            from_lines_path=from_headers_path,
            to_lines_path=to_headers_path,
            start_line_index=header_start_line_index,
            end_line_index=header_end_line_index,
            need_translate=need_translate,
            need_copy_translation=need_copy_translation,
            max_workers=max_workers,
            use_headless=use_headless,
            wait_query=wait_query,
        )

    # make texts
    out_texts = mk_texts(to_language)

    if need_write_texts:
        write_texts(
            from_texts_path=from_texts_path,
            to_sentences_path=to_sentences_path,
            to_headers_path=to_headers_path,
            to_texts_path=out_texts,
        )


def main():
    start_time = time.time()

    # prepare directories
    data_dir.mkdir(parents=True, exist_ok=True)
    subdirs = [
        sentences_dir,
        headers_dir,
        decks_dir,
        texts_dir,
        progress_dir,
    ]
    for i in subdirs:
        (data_dir / i).mkdir(parents=True, exist_ok=True)

    # download tokenizer
    need_download_tokenizer = False

    if need_download_tokenizer:
        nltk.download("punkt")

    # fetch and write texts in English
    need_fetch_texts = False

    en_language = "en"
    en_texts_path = mk_texts(en_language)
    if need_fetch_texts:
        write_texts_en(
            texts_path=en_texts_path,
        )

    # Prepare English files for translation
    need_write_data = False

    en_headers_path = mk_headers(en_language)
    en_sentences_path = mk_sentences(en_language)

    if need_write_data:
        write_data_en(
            headers_path=en_headers_path,
            sentences_path=en_sentences_path,
            texts_path=en_texts_path,
        )

    # Translate files
    need_full_translate = False
    need_translate = False
    need_translate_sentences = True
    need_translate_headers = True
    need_copy_translation = True
    need_write_texts = True
    use_headless = False
    wait_query = 30
    max_workers = 10

    languages = ["ru", "de", "es", "it"]
    
    
    if need_full_translate:
        for language in languages:
            full_translate(
                from_language=en_language,
                to_language=language,
                from_sentences_path=en_sentences_path,
                from_headers_path=en_headers_path,
                from_texts_path=en_texts_path,
                need_translate_sentences=need_translate_sentences,
                need_translate_headers=need_translate_headers,
                max_workers=max_workers,
                use_headless=use_headless,
                sentence_start_line_index=1,
                header_start_line_index=1,
                sentence_end_line_index=25000,
                header_end_line_index=1500,
                need_copy_translation=need_copy_translation,
                need_translate=need_translate,
                need_write_texts=need_write_texts,
                wait_query=wait_query,
            )
    
    # Write decks
    need_write_decks = True

    if need_write_decks:
        languages_all = [en_language] + languages
        for i, from_language in enumerate(languages_all):
            for to_language in languages_all[i + 1 :]:
                print(f"writing deck: {from_language} <-> {to_language}")
                deck_path = mk_deck(from_language, to_language)
                write_deck(
                    from_texts_path=mk_texts(from_language),
                    to_headers_path=mk_headers(to_language),
                    to_sentences_path=mk_sentences(to_language),
                    deck_path=deck_path,
                )

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    for i in range(100):
        try:
            main()
            break
        except Exception as e:
            print(e)
            continue

#%%
# generate deck titles

# import pandas as pd

# df = pd.read_csv("data/translations.csv", sep=";", encoding="utf8", index_col=0)
# #%%
# langs = ["en", "ru", "de", "es", "it"]
# stories = "stories"
# # df["en"]
# # df.index
# # df.loc[["en"]]
# for i,l1 in enumerate(langs):
#     for l2 in langs[i+1:]:
#         print(f"{df[l1][l1]}-{df[l2][l1]} / {df[l1][l2]}-{df[l2][l2]} / {df[stories][l1]}-{df[stories][l2]}")
# en = {
#     "Italian": "Italiano"
# }

#%%
# check progress blocks


# from bs4 import BeautifulSoup

# with open("data/progress/en-de-1-21678.csv", "r", encoding="utf8") as short:
#     with open("data/progress/en-ru-1-21678.csv", "r", encoding="utf8") as long:
#         soup_enes = BeautifulSoup(short.read(),"html.parser")
#         soup_enru = BeautifulSoup(long.read(),"html.parser")
#         lengths_enes = {
#             x.find("index"): len(x.find("body").text.strip().split("\n"))
#             for x in soup_enes.find_all("block")
#         }
#         lengths_enru = {
#             x.find("index"): len(x.find("body").text.strip().split("\n"))
#             for x in soup_enru.find_all("block")
#         }
#         for k, v in lengths_enes.items():
#             if lengths_enru[k] != v:
#                 print(k)

# %%
