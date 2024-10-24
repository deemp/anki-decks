# %%

# Download list of English words together with topics and image URLs

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4.element import Tag
from os.path import basename
import os
from selenium.webdriver.chrome.webdriver import WebDriver
import pandas as pd
from selenium.webdriver.common.by import By

os.chdir(f'{os.environ["ROOT_DIR"]}/En-Ru-Picture/words')

options = Options()
options.headless = True
options.add_argument(f"--blink-settings=imagesEnabled=false")

encoding = "UTF-8"

out = "./out"
mkPath = lambda f: f"{out}/{f}.csv"

f_word_data_en = mkPath("word_data_en")
f_topic_without_words_urls = mkPath("topic_without_words_urls")
f_all_topic_urls = mkPath("all_topic_urls")
f_remaining_topic_urls = mkPath("remaining_topic_urls")
f_topic_with_words_urls = mkPath("topic_with_words_urls")
f_problematic_topic_urls = mkPath("problematic_topic_urls")

htmlParser = "html.parser"


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


get_topic_urls()
init_remaining_topic_urls()

while True:
    if not exist_remaining_topic_urls():
        break

# if __name__ == "__main__":
#     print("\n\n")
#     print (f"Words: {f_words}")
#     print (f"All URLs: {f_all_urls}")
#     print (f"Remaining URLs (to be fetched): {f_remaining_urls}")
#     print (f"Problematic URLs: {f_problems}")
#     print (f"URLs without words: {f_no_words}")
#     print (f"URLs with words: {f_has_words}")
#     print("\n\n")

#     os.makedirs(out,exist_ok=True)


# %%

mkText = lambda f: f"{out}/{f}.txt"

en_tag = "en_tag"
en_word = "en_word"
img_link = "img_link"
img_file = "img_file"
d_images = f'{out}/images'
f_deck = mkText("deck")
img_html = "img_html"
f_en_words = mkText("en_words")
f_en_tags = mkText("en_tags")
ru_tag = "ru_tag"
f_ru_tags = mkText("ru_tags")
f_ru_words = mkText("ru_words")
ru_word = "ru_word"
f_anki = mkText("anki_deck")
f_deck_edited = mkText("deck_edited")

#%%

def produce_file_names():
    df = pd.read_csv(f_words, header=None)
    df.columns = [en_tag, en_word, img_link]
    for i in df.columns:
        df[i]=df[i].str.strip()
    # adjust file names
    df[img_file] = df[en_word]
    df[img_file] = (df[img_file].str.replace('-','_').str.strip().str.replace(' ', '_') + ".jpg").str.strip()

    return df

df1 = produce_file_names()
df1.head(5)
# %%

# download images
import requests
from concurrent.futures import ThreadPoolExecutor


def save_img(url, file):
    print(file)
    response = requests.get(url).content
    with open(f'{d_images}/{file}', 'wb') as handle:
        handle.write(response)

def load_images(df):
    os.makedirs(d_images, exist_ok=True)
    with ThreadPoolExecutor(max_workers=100) as executor:
        # TODO check
        executor.map(save_img, df[img_link], df.iloc[img_file])


load_images(df1)

# %%

def create_links(df):
    df[img_html] = df[img_file]
    for i in range(df.shape[0]):
        df[img_html].iloc[i] = f'<img src="{df[img_file].iloc[i]}">'
    return df

df2 = create_links(df1)
df2.head()
#%%


#%%

def write_english_words(df):
    df[en_word].to_csv(f_en_words, index=False, header=False)

write_english_words(df2)
# %%

# export tags
import numpy as np


def write_en_tags(df):
    np.savetxt(f_en_tags, df[en_tag].str.strip().unique(), fmt="%s")

write_en_tags(df2)

# next, translate the tags and paste them into `ru_tags`

#%%
# add russian tags

def add_russian_tags(df):
    df_en = pd.read_csv(f_en_tags, header=None)
    df_en.columns = [en_tag]
    
    df_ru = pd.read_csv(f_ru_tags, header=None)
    df_ru.columns = [ru_tag]
    df_en_ru = pd.concat([df_en, df_ru], axis=1)

    return pd.merge(df, df_en_ru, how='left', on=en_tag)

df3 = add_russian_tags(df2)
df3
# %%

# now, copy the english words into Deepl
# paste the translations into a file `ru_words``
# create the final file

def combine(df):
    ru_df = pd.read_csv(f_ru_words, header=None)
    ru_df.columns = [ru_word]
    comb = pd.concat([df[en_word], ru_df, df[[img_html, en_tag, ru_tag]]], axis=1)
    return comb

df4 = combine(df3)
df4

# %%

# export the final deck

df4.to_csv(f_deck, sep=";", index=False, header=False, quoting=3)
# %%

# edit in anki, merge changes from `anki_deck`


anki_export_columns = [en_word, ru_word]
deck_columns = [en_word, ru_word, img_html, en_tag, ru_tag]

def edit_via_anki():
    df_deck = pd.read_csv(f_deck, header=None, sep=";")
    df_deck.columns = deck_columns
    df_anki = pd.read_csv(f_anki, header=None, sep=";")
    df_anki.columns = anki_export_columns
    df_deck.merge(df_anki, on=en_word)    
    return df_deck

df5 = edit_via_anki()
df5
# %%

# save edited deck

df5.to_csv(f_deck_edited, sep="|", index=False, header=False, quoting=3)

# %%

# now, we can load the edited deck into anki!