# Anki decks

A collection of [Anki](https://apps.ankiweb.net/) decks that can be used with Google TTS on Android. Also shared [here](https://ankiweb.net/shared/byauthor/1890287529).

## Pairs

* [De-Ru](./De-Ru/de-ru.md)
* [De-Ru (Phrases)](./De-Ru-phrases/de-ru-phrases.md)
* [Fr-Ru](./Fr-Ru/fr-ru.md)
* [Uz-Ru](./Uz-Ru/uz-ru.md)
* [En-Ru](./En-Ru/en-ru.md)
* [Es-Ru](./Es-Ru/es-ru.md)
* [En-Ru-Picture](./En-Ru-Picture/README.md)

## Wiki

### Tools

* [VS Code](https://code.visualstudio.com/)

* VS Code extensions
  * [LTeX](https://marketplace.visualstudio.com/items?itemName=valentjn.vscode-ltex)
    * Grammar checking
  * [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv)
    * SQL queries
    * Column alignment

* [GoldenDict](https://t.me/goldendict)
  * Manual grammar checking

### Audio

1. Choose the length of pauses between phrases
1. Record phrases
1. Convert to `mp3` and [split](https://unix.stackexchange.com/questions/318164/sox-split-audio-on-silence-but-keep-silence).

    ```console
    ffmpeg -v 5 -y -i from.m4a -acodec libmp3lame -ac 2 -ab 192k to.mp3
    mkdir out
    sox to.mp3 out/out_.mp3 silence -l 1 0.2 0.02% 1 0.3 0.2% : newfile : restart
    ```
