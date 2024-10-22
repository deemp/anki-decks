# Create a Deck

If you need to study a set of words, you may want to create a [deck](https://docs.ankiweb.net/getting-started.html#decks) for them.

In this article, I describe an algorithm for generating data for a German-English deck, given a list of German words.

## Given

A list of German words in a one-column [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file. Example:

  ```text
  Sonne
  paukt
  meinung
  starke
  ```

A list of words can be extracted from a text using ChatGPT.

## Goal

Generate a CSV file with data suitable for [importing](https://docs.ankiweb.net/importing/text-files.html) into Anki.

The CSV file should contain the following columns:

1. `index` - Index of the row.
1. `part_of_speech` - Part of speech of the German word (lowercase).
1. `word_de` - The German word.
1. `word_forms_de` - Forms of the German word:
    - For nouns: the plural form with articles, if applicable.
    - For verbs: principal parts.
    - For adjectives: degrees (positive, comparative, superlative), if applicable.
1. `sentence_de` - A sentence in German using the word in the specified part of speech.
    - The sentence should be 30 to 40 characters long.
1. `sentence_en` - The English translation of the German word:
    - For nouns: a word without an article.
    - For verbs: the infinitive form.
1. `word_en` - The translation of the German sentence into English.

## Algorithm

1. Create a list of German words in a CSV file. Example:
  
    ```text
    Sonne
    paukt
    meinung
    starke
    ```

1. Ask ChatGPT to write the words in their lemmatized form.

    ```text
    ## Task

    Write the input German words in their lemmatized form.

    ## Lemmatized Form

    - Nouns should include articles.
    - Adjectives must be in the masculine singular form.
    - Verbs should be in the infinitive form.

    ## Output

    - Use "|" as a separator.
    - Include a header.
    - Output as a CSV Markdown code block.

    ## Columns

    1. `index` - Index of the row.
    2. `part_of_speech` - Part of speech.
    3. `word_de` - Lemmatized form of the German word.

    ## Input

    Sonne
    paukt
    meinung
    starkes
    ```

1. The output should be:

    ```text
    index|part_of_speech|word_de
    1|noun|die Sonne
    2|verb|pauken
    3|noun|die Meinung
    4|adjective|stark
    ```
  
1. Ask ChatGPT to extend the rows:
  
    ```text
    ## Task

    Extend the rows by creating additional columns.

    ## Columns

    1. `index` - Index of the row.
    2. `part_of_speech` - Part of speech of the German word (lowercase).
    3. `word_de` - The German word.
    4. `word_forms_de` - Forms of the German word:
        - For nouns: the plural form with articles, if applicable.
        - For verbs: principal parts.
        - For adjectives: degrees (positive, comparative, superlative), if applicable.
    5. `sentence_de` - A sentence in German using the word in the specified part of speech.
        - The sentence should be 30 to 40 characters long.
    6. `sentence_en` - The English translation of the German word:
        - For nouns: a word without an article.
        - For verbs: the infinitive form.
    7. `word_en` - The translation of the German sentence into English.
    8. `sentence_repeated_de` - The same sentence as in `sentence_de`, used for sorting by the length of the German sentence.

    ## Output

    - Use "|" as a separator.
    - Include a header.
    - Output as a CSV Markdown code block.

    ## Input
    
    index|part_of_speech|word_de
    1|noun|die Sonne
    2|verb|pauken
    3|noun|die Meinung
    4|adjective|stark
    ```

1. The output should be:

    ```text
    index|part_of_speech|word_de|word_forms_de|sentence_de|word_en|sentence_en|sentence_repeated_de
    1|noun|die Sonne|die Sonnen|Die Sonne scheint heute sehr hell.|sun|The sun is shining very brightly today.|Die Sonne scheint heute sehr hell.
    2|verb|pauken|pauken, paukte, hat gepaukt|Ich muss für die Prüfung pauken.|to cram|I have to cram for the exam.|Ich muss für die Prüfung pauken.
    3|noun|die Meinung|die Meinungen|Meine Meinung ist anders als deine.|opinion|My opinion is different from yours.|Meine Meinung ist anders als deine.
    4|adjective|stark|stärker, am stärksten|Der Wind weht heute sehr stark.|strong|The wind is blowing very strongly today.|Der Wind weht heute sehr stark.
    ```

1. Identify and mark problematic rows. Possible issues include:
    - `sentence_de` is not between 30 and 40 characters long.
    - `sentence_de` is not a complete sentence.
    - `sentence_de` provides an "uninteresting" context for `word_de`.
    - `word_de` is an adjective but is not used as one in `sentence_de`.
    - `sentence_de` does not contain `word_de`.
    - `sentence_en` does not contain `word_en`.

1. Partition the rows into "good" and "problematic."
    - Use Excel, Google Sheets, LibreOffice Calc, etc.
    - Use [VS Code](https://code.visualstudio.com/):
        1. Copy the output into a CSV file.
        2. Open the file in VS Code and align columns using the [rainbow-csv](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) extension.
        3. Sort rows by length using the [sort-lines](https://marketplace.visualstudio.com/items?itemName=Tyriar.sort-lines) extension.
        4. Move the problematic rows to the end of the file.
    - Use [pandas](https://pandas.pydata.org/) if you prefer scripting tools.

1. Ask ChatGPT to extract only the index, part of speech, and German word from problematic rows. Example:

    ```text
    ## Task

    Extract only the first 3 columns (index, part of speech, and the German word).

    ## Output

    - Use "|" as a separator.
    - Include a header.
    - Output as a CSV Markdown code block.

    ## Input

    index|part_of_speech|word_de|word_forms_de|sentence_de|word_en|sentence_en|sentence_repeated_de
    1|noun|die Sonne|die Sonnen|Die Sonne scheint heute sehr hell.|sun|The sun is shining very brightly today.|Die Sonne scheint heute sehr hell.
    2|verb|pauken|pauken, paukte, hat gepaukt|Ich muss für die Prüfung pauken.|to cram|I have to cram for the exam.|Ich muss für die Prüfung pauken.
    3|noun|die Meinung|die Meinungen|Meine Meinung ist anders als deine.|opinion|My opinion is different from yours.|Meine Meinung ist anders als deine.
    4|adjective|stark|stärker, am stärksten|Der Wind weht heute sehr stark.|strong|The wind is blowing very strongly today.|Der Wind weht heute sehr stark.
    ```

1. The output should be:

    ```text
    index|part_of_speech|word_de
    1|noun|die Sonne
    2|verb|pauken
    3|noun|die Meinung
    4|adjective|stark
    ```

1. Ask ChatGPT to regenerate the rows if necessary.
1. Repeat the generation and filtering process until all rows are acceptable.
1. Sort the rows by `index`.
1. Save all acceptable rows into a CSV file.
