# %%

# Download list of English words together with topics and image URLs.

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4.element import Tag
from os.path import basename
import os
from selenium.webdriver.chrome.webdriver import WebDriver
import pandas as pd
from selenium.webdriver.common.by import By
from IPython.display import display
import requests
from concurrent.futures import ThreadPoolExecutor
import json


os.chdir(f'{os.environ["ROOT_DIR"]}/En-Ru-Picture/words')

encoding = "UTF-8"

out = "./out"
mk_path = lambda f: f"{out}/{f}.csv"

f_word_data_en = mk_path("word_data_en")
f_topic_without_words_urls = mk_path("topic_without_words_urls")
f_all_topic_urls = mk_path("all_topic_urls")
f_remaining_topic_urls = mk_path("remaining_topic_urls")
f_topic_with_words_urls = mk_path("topic_with_words_urls")
f_problematic_topic_urls = mk_path("problematic_topic_urls")

htmlParser = "html.parser"

f_topics_en = mk_path("topics_en")
f_words_en = mk_path("words_en")

f_topics_ru = mk_path("topics_ru")
f_words_ru = mk_path("words_ru")

column_topic_en = "topic_en"
column_word_en = "word_en"

column_topic_ru = "topic_ru"
column_word_ru = "word_ru"

column_image_url = "image_url"


column_image_file = "image_file"
images_dir = f"{out}/images"

f_deck = mk_path("../deck")

# %%

options = Options()
options.headless = True
options.add_argument(f"--blink-settings=imagesEnabled=false")


def get_topic_words(url):
    """get words for a topic"""

    chromedriver = webdriver.Chrome(options=options, keep_alive=False)
    with chromedriver as driver:
        driver: WebDriver

        driver.get(url)
        soup = BeautifulSoup(driver.page_source, htmlParser)

        topic = soup.find("h1", attrs={"class": "page-header"}).text.strip()

        # check contains words
        content = driver.find_element(by=By.CLASS_NAME, value="content")
        try:
            context = content.find_element(
                by=By.XPATH, value="./div[contains(@class, 'contextual-region')]"
            )

        except Exception as ex:
            print("No words detected!")
            with open(f_topic_without_words_urls, "a", encoding=encoding) as f:
                f.write(url)
        else:
            # take the words

            iframe = context.find_element(by=By.TAG_NAME, value="iframe")

            driver.switch_to.frame(iframe)

            iframe1 = driver.find_element(by=By.TAG_NAME, value="iframe")

            driver.switch_to.frame(iframe1)

            soup = BeautifulSoup(driver.page_source, htmlParser)

            qs = soup.find("div", attrs={"id": "questions"})
            qlist = qs("div", attrs={"class": "question"}, recursive=True)

            with open(f_word_data_en, "a", encoding=encoding) as f:
                for q in qlist:
                    q: Tag

                    src = f"https:{str(q.find('img')['src'])}"
                    ans = q.find("li")(text=True, recursive=False)[0]
                    f.write(f"{topic}|{ans}|{src}\n")


def get_topic_urls():
    """fetch a list of topic URLs"""
    chromedriver = webdriver.Chrome(options=options, keep_alive=False)
    with chromedriver as driver:
        top_url = (
            "https://learnenglishkids.britishcouncil.org/grammar-vocabulary/word-games"
        )
        print(f"Fetching all urls from {top_url}")
        driver.get(top_url)
        body = driver.find_element(by=By.TAG_NAME, value="body").get_attribute(
            "innerHTML"
        )
        sec = BeautifulSoup(body, htmlParser).find(
            "section",
            attrs={"id": "block-views-block-magazine-glossary-block-magazine-glossary"},
        )

        div = sec("div", attrs={"class": "view-content"}, recursive=True)[1]
        divs = div("div", attrs={"class": "views-row"})

        with open(file=f_all_topic_urls, mode="a", encoding=encoding) as f:
            for d in divs:
                d: Tag = d
                ref = basename(
                    d.find("a", attrs={"rel": "bookmark"}, recursive=True)["href"]
                )
                src = f"{top_url}/{ref}"
                f.write(f"{src}\n")
            f.write("\n")


def get_remaining_topic_words():
    no_problems = True
    with open(file=f_remaining_topic_urls, mode="r", encoding=encoding) as f:
        urls = f.readlines()
        for url in urls:
            print(f"Fetching {url}")
            try:
                get_topic_words(url)
            except Exception as e:
                print(e)
                with open(
                    file=f_problematic_topic_urls, mode="a", encoding=encoding
                ) as f:
                    f.write(url)
                no_problems = False
            else:
                print("Ok")
                with open(
                    file=f_topic_with_words_urls, mode="a", encoding=encoding
                ) as f:
                    f.write(url)
    return no_problems


def init_remaining_topic_urls():
    """copy the list of all topic urls into the list of remaining topic urls"""
    with open(file=f_all_topic_urls, mode="r", encoding=encoding) as all_urls:
        with open(
            file=f_remaining_topic_urls, mode="w", encoding=encoding
        ) as remaining_urls:
            remaining_urls.write(all_urls.read())


def exist_remaining_topic_urls():
    if get_remaining_topic_words():
        with open(
            file=f_remaining_topic_urls, mode="w", encoding=encoding
        ) as remaining_urls:
            remaining_urls.write("")
        with open(
            file=f_problematic_topic_urls, mode="w", encoding=encoding
        ) as remaining_urls:
            remaining_urls.write("")
        return False

    with open(file=f_problematic_topic_urls, mode="r", encoding=encoding) as problems:
        with open(
            file=f_remaining_topic_urls, mode="w", encoding=encoding
        ) as remaining_urls:
            remaining_urls.write(problems.read())
    with open(file=f_problematic_topic_urls, mode="w", encoding=encoding) as problems:
        problems.write("")

    return True


def get_data():
    get_topic_urls()
    init_remaining_topic_urls()

    while True:
        if not exist_remaining_topic_urls():
            break


# Uncomment to get English word data

# get_data()

# %%

# Extract English words.


def process_data():
    word_data_en = pd.read_csv(f_word_data_en, sep="|", header=None)
    word_data_en.columns = [column_topic_en, column_word_en, column_image_url]

    word_data_en["word_en"].map(lambda x: x.strip()).to_csv(
        f_words_en, header=False, sep="|"
    )

    topics_en = word_data_en["topic_en"].map(lambda x: x.strip()).unique()
    pd.DataFrame(topics_en).to_csv(f_topics_en, header=False, sep="|")


process_data()

# %%


def produce_file_names():
    df = pd.read_csv(f_word_data_en, header=None, sep="|")
    df.columns = [column_topic_en, column_word_en, column_image_url]
    for i in df.columns:
        df[i] = df[i].str.strip()
    # adjust file names
    df[column_image_file] = df[[column_topic_en, column_word_en]].apply(
        lambda x: f"{x[column_topic_en]}___{x[column_word_en]}".lower(), axis=1
    )
    df[column_image_file] = (
        df[column_image_file].str.strip().str.replace(" ", "-") + ".jpg"
    ).str.strip()
    df[column_image_file] = df[column_image_file].map(lambda x: f"en-ru-pictures___{x}")

    return df


df_with_image_file_names = produce_file_names()
df_with_image_file_names.head()

# %%

# Download images.

# 1. Open an image by URL.
# 2. Check Network in Dev tools.
# 3. Copy cookies and write as JSON.
# 4. Copy the "User-Agent" header and write as JSON.


with open("cookies.json", "r") as file:
    cookies = json.load(file)

with open("headers.json", "r") as file:
    headers = json.load(file)


def save_img(url, file):
    response = requests.get(
        url=url, timeout=30, cookies=cookies, headers=headers
    ).content
    with open(f"{images_dir}/{file}", "wb") as handle:
        handle.write(response)
    print(f"Downloaded {url} and wrote to {file}")


def download_images(df: pd.DataFrame):
    os.makedirs(images_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=100) as executor:
        # TODO check
        executor.map(save_img, df[column_image_url], df[column_image_file])


download_images(df_with_image_file_names)

# %%

column_image_html = "image_html"


def add_image_html(df: pd.DataFrame):
    df[column_image_html] = df[column_image_file].map(lambda x: f'<img src="{x}">')
    df.drop(labels=[column_image_file, column_image_url], axis=1, inplace=True)
    return df


df_with_image_html = add_image_html(df_with_image_file_names)
df_with_image_html.head()


# %%
# Add Russian tags.


def add_russian_tags(df: pd.DataFrame):
    df_ru = pd.read_csv(f_topics_ru, header=None, index_col=0, sep="|")
    df_en = pd.read_csv(f_topics_en, header=None, index_col=0, sep="|")

    df_en.columns = [column_topic_en]
    df_ru.columns = [column_topic_ru]
    df_en_ru = pd.concat([df_en, df_ru], axis=1)

    return pd.merge(df, df_en_ru, how="left", on=column_topic_en)


df_with_topics_ru = add_russian_tags(df_with_image_html)
df_with_topics_ru.head()
# %%

# Add Russian words.


def add_russian_words(df: pd.DataFrame):
    df_ru = pd.read_csv(f_words_ru, header=None, index_col=0, sep="|")
    df_ru.columns = [column_word_ru]
    return pd.concat([df, df_ru], axis=1)


df_with_words_ru = add_russian_words(df_with_topics_ru)
df_with_words_ru.head()

# %%

# Reorder columns


def reorder_columns(df: pd.DataFrame):
    df = df[
        [
            column_word_en,
            column_word_ru,
            column_topic_en,
            column_topic_ru,
            column_image_html,
        ]
    ]
    return df


df_reordered = reorder_columns(df_with_words_ru)
df_reordered.head()

# %%

# Compose the deck

df_reordered.to_csv(f_deck, sep="|", quoting=3)
