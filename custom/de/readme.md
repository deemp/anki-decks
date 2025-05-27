# Custom deck

This deck provides cards for studying specific unknown German words from given texts.

Cards are given in the CSV format in the "raw deck" [deck.csv](./deck.csv).

Each card contains:

1. an index;
1. a German word;
1. a part of speech of the German word (2);
1. a translation of the German word (2) to English;
1. a German sentence that contains the German word (2);
1. a translation of the German sentence (5) to English that contains the translation of the German word (4).
1. a lemmatized version of the German sentence.

    > The goal of lemmatization is to reduce a word to its root form, also called a lemma. For example, the verb `running` would be identified as `run`.
    >
    > [src](https://www.techtarget.com/searchenterpriseai/definition/lemmatization)

## Files

### Data

- `./`
  - [`deck.csv`](./deck.csv) - "raw deck", a `|`-separated CSV with cards data followed by a list of indexed German words that don't yet have data.
- `./data/`
  - `external/`
    - [`dewiki-noun-articles.csv`](data/external/dewiki-noun-articles.csv) - nouns with articles from [dewiktionary](https://github.com/deemp/german-nouns?tab=readme-ov-file#2-compile-the-list-of-nouns-from-a-wiktionary-xml-file). Can be downloaded [here](https://github.com/deemp/german-nouns/blob/main/german_nouns/nouns.csv).
    - [`dwds_lemmata_2025-01-15.csv`](data/external/dwds_lemmata_2025-01-15.csv) - lemmata from DWDS. Can be downloaded [here](https://www.dwds.de/lemma/list#download).
  - `sources/`
    - `playlist/`
      - [`raw.csv`](data/sources/playlist/raw.csv) - a list of song text titles and their authors.
      - [`data.yaml`](data/sources/playlist/data.yaml) - a list of objects that describe texts. The `i`-th object in the list has the index `i` in [`data.csv`](data/sources/playlist/data.csv).
      - [`data.csv`](data/sources/playlist/data.csv) - song texts in CSV format.
    - [`lemmatized.csv`](data/sources/lemmatized.csv) - almost the same as [`playlist/data.csv`](data/sources/playlist/data.csv), but the texts are lemmatized.
    - [`words.csv`](data/sources/words.csv) - indexed words from all texts, in the order as they appear in the lemmatized texts in [`lemmatized.csv`](data/sources/lemmatized.csv).

### Scripts

- `./script/`
  - [`lib.py`](./script/lib.py) - mostly language-agnostic functionality for generating cards given the "raw deck".
  - [`main.py`](./script/main.py) - provides deck-specific functionality for fetching song texts, lemmatization, extracting words, collecting unknown words, preparing the "raw deck", and running the cards generator.
  - [`api_request_parallel_processor.py`](./script/api_request_parallel_processor.py) - used for sending parallel requests to the OpenAI API.

## Setup

Tested on my Linux machine.

- Install the Nix package manager ([link](https://nixos.org/)) and reload the computer.
  - [method 1](https://github.com/DeterminateSystems/nix-installer#determinate-nix-installer).
  - [method 2](https://nixos.org/download/) + [enable flakes](https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager) permanently - I prefer the single-user installation because it's easier to manage;
- Install [direnv](https://direnv.net/docs/installation.html) (don't forget to hook it into your shell!).
- Clone this repository.

  ```console
  git clone https://github.com/deemp/anki-decks
  ```
  
- Open VS Code in the repository directory.

  ```console
  code anki-decks
  ```

- Install the recommended VS Code extensions.
  - You can open the Command Palette (`Ctrl + Shift + P` on Linux), type and click `Extensions: Show Recommended Extensions` or `Configure Recommeded Extensions`.

- Open the terminal (<code>Ctrl + \`</code> on Linux) and allow `direnv` to work in the repository directory.

  ```console
  direnv allow
  ```

- Answer `yes` to questions.

- Install Python dependencies.

  ```console
  nix develop
  poetry install
  poetry run python -m spacy download de_core_news_lg
  ```

- Open the Command Palette, type and click `direnv: Reset and reload environment`.

- If you plan to edit Nix files, e.g. [flake.nix](../../flake.nix), install `nil`.

  ```console
  nix profile install nixpkgs#nil
  ```

- Open the Command Palette, type and click `Python: Select Interpreter`. Click the option that has `./.venv/bin/python`.

- Open [main.py](./script/main.py). You should see the `Run cell` buttons above the `# %%` comments.

- Create a `.env` file in the root directory of the repository with your credentials for [OpenAI](https://openai.com/) and [Genius.com](https://genius.com/).

  ```console
  OPENAI_API_KEY=
  GENIUS_CLIENT_ID=
  GENIUS_CLIENT_SECRET=
  GENIUS_CLIENT_ACCESS_TOKEN=
  ```

## Usage

- In [main.py](./script/main.py), go to the cell containing `nlp = spacy.load ...` and run the cell (`Ctrl + Enter`)
