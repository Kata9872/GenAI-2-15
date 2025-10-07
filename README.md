# Генерация текста с ограничением повторов
Этот скрипт сравнивает генерацию текста моделью GPT-2 в двух режимах:
##### Baseline (без ограничения): 
обычная генерация (может повторяться)
##### С ограничением: 
с параметром no_repeat_ngram_size=2, запрещающим повторение любых пар слов
###### Цель — доказать, что ограничение снижает количество повторяющихся триграмм как минимум на 30%.

### Как запустить
Установите зависимости:
```bash
pip install -r requirements.txt
```
### Запустите скрипт с желаемой длиной генерации (в токенах):
```bash
python GenAI-2-15.py 30 50
```
### Введите текстовый промпт при запросе enter prompt:
###### Пример промпта:
```text
I love chocolate. I love chocolate. I love
```
###### Программа выведет:
```text
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Без ограничения: I love chocolate. I love chocolate. I love chocolate.

So I know what you're thinking. Chocolate is for you.

So, I'm gonna try my best to keep you in line with the way I want to tell you about me.

I'm gonna try to
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
С ограничением: I love chocolate. I love chocolate. I love the way it smells. It's not really chocolate but it's what I'm looking for. The way I see it is that chocolate is just what you want. In that sense, it makes me happy."

The two went on to form

Повторяющихся триграмм — baseline: 7
Повторяющихся триграмм — с ограничением: 3
Снижение повторов: 57.1%
Условие выполнено. Снижение ≥30%
```

#### Требования
Python 3.7+
#### Библиотеки из requirements.txt:
```text
transformers>=4.30
torch
```
