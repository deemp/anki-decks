# %%

import os

os.chdir(f'{os.environ["ROOT_DIR"]}/frequency/de')


def mk_template_path(x: str):
    return f"de-en/anki-deck/audio-files/ankiweb-description{x}.html"


def read_template():
    with open(mk_template_path(""), "r", encoding="UTF-8") as t:
        return t.read()


template = read_template()


def write_template(template_path_suff: str, replacement: str):
    with open(mk_template_path(template_path_suff), "w", encoding="UTF-8") as t:
        t.write(template.replace("{{n_words}}", replacement))


write_template("-1", "the 1st to the 3000th")
write_template("-2", "the 3001st to the last")
