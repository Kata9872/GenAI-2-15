# Установите зависимости: pip install -r requirements.txt
# Рекомендуемая версия transformers >= 4.30 (для корректной работы text-generation pipeline)
from transformers import pipeline, AutoTokenizer
import argparse
import sys
import importlib
from collections import Counter

# Проверка наличия и версии библиотеки transformers
try:
    transformers = importlib.import_module("transformers")
    required_version = "4.34"
    current_version = transformers.__version__

    # Сравниваем версии
    if tuple(map(int, current_version.split('.')[:2])) < tuple(map(int, required_version.split('.'))):
        print(f"Требуется transformers>={required_version}, у вас {current_version}")
        print("Обновите: pip install --upgrade transformers")
        sys.exit(1)

except ImportError:
    print("Библиотека 'transformers' не установлена!")
    print("Установите: pip install transformers torch")
    sys.exit(1)


def generate_text(generator, prompt, min_length, max_length, no_repeat_ngram_size=None):
    """
    Генерирует текст с использованием модели GPT-2 через Hugging Face pipeline.
    Args:
        generator (transformers.Pipeline): Инициализированный пайплайн генерации текста.
        prompt (str): Начальный текст (промпт) для генерации.
        min_length (int): Минимальное количество новых токенов для генерации.
        max_length (int): Максимальное количество новых токенов для генерации.
        no_repeat_ngram_size (int, optional): Если задано, запрещает повторение n-грамм указанного размера.
    Returns:
        list or None: Список с результатом генерации (словарь с ключом 'generated_text') или None при ошибке.
    Raises:
        ValueError: Если входные параметры имеют неверный тип или значение.
    """
    try:
        kw_args = { # keyword arguments
            "min_new_tokens": min_length,
            "max_new_tokens": max_length,
            "do_sample": True,        # включает случайную генерацию (не детерминированную)
            "temperature": 0.9,       # регулирует "креативность", свободу
            "top_p": 0.9,             # берёт токены из верхних 90% вероятностей
            "truncation": True        # обрезает длинный вход, если превышает лимит
        }
        if no_repeat_ngram_size is not None:
            kw_args["no_repeat_ngram_size"] = no_repeat_ngram_size

        result = generator(prompt, **kw_args)
        return result

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None


def count_duplicate_trigrams(text, tokenizer):
    """
    Подсчитывает количество повторяющихся триграмм (3-грамм) в тексте.
    Args:
        text (str): Текст для анализа.
        tokenizer (transformers.PreTrainedTokenizer): Токенизатор для преобразования текста в токены.
    Returns:
        int: Количество уникальных триграмм, встречающихся более одного раза.
    Raises:
        ValueError: Если входные параметры имеют неверный тип.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    counts = Counter(trigrams)
    return sum(1 for c in counts.values() if c > 1)


def main():
    """
    Основная функция программы для сравнения генерации текста с и без ограничения на повторы.
    Программа:
    1. Принимает min_length и max_length как аргументы командной строки
    2. Запрашивает у пользователя текстовый промпт
    3. Генерирует два варианта текста (с ограничением no_repeat_ngram_size=2 и без)
    4. Подсчитывает повторяющиеся триграммы в каждом
    5. Выводит сравнение и проверяет условие снижения повторов ≥30%
    Raises:
        ValueError: При некорректных входных аргументах или данных.
        SystemExit: При отсутствии необходимых библиотек.
    """
    parser = argparse.ArgumentParser(description="Сравнение генерации текста с и без ограничения на повторы")
    parser.add_argument("min_length")
    parser.add_argument("max_length")
    args = parser.parse_args()

    try:
        min_length = int(args.min_length)
        max_length = int(args.max_length)
    except ValueError:
        raise ValueError("min_length и max_length должны быть целыми числами")

    # Валидация входных данных
    if min_length <= 0:
        raise ValueError("min_length должно быть положительным")
    if max_length <= 0:
        raise ValueError("max_length должно быть положительным")
    if min_length > max_length:
        raise ValueError("min_length не может быть больше max_length")
    if max_length > 1024:
        print("Предупреждение: GPT-2 поддерживает максимум 1024 токена. Обрезаем до 1024.")
        max_length = 1024

    # Инициализация модели и токенизатора
    try:
        generator = pipeline("text-generation", model="gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Не удалось загрузить модель GPT-2: {e}")
        print("Возможные причины: нет интернета, мало памяти, ошибка совместимости.")
        sys.exit(1)

    print("Введите промпт:\n")
    prompt = input("enter prompt: ")

    # БЕЗ no_repeat_ngram_size (baseline)
    result_baseline = generate_text(generator, prompt, min_length, max_length, no_repeat_ngram_size=None)
    if result_baseline is None:
        print("Не удалось сгенерировать baseline")
        return
    text_baseline = result_baseline[0]["generated_text"]
    print("Без ограничения:", text_baseline)

    # С no_repeat_ngram_size=2
    result_no_repeat = generate_text(generator, prompt, min_length, max_length, no_repeat_ngram_size=2)
    if result_no_repeat is None:
        print("Не удалось сгенерировать с ограничением")
        return
    text_no_repeat = result_no_repeat[0]["generated_text"]
    print("С ограничением:", text_no_repeat)

    # Подсчёт и сравнение триграмм
    dup_base = count_duplicate_trigrams(text_baseline, tokenizer)
    dup_nr = count_duplicate_trigrams(text_no_repeat, tokenizer)
    print(f"\nПовторяющихся триграмм — baseline: {dup_base}")
    print(f"Повторяющихся триграмм — с ограничением: {dup_nr}")

    if dup_base > 0:
        reduction = (dup_base - dup_nr) / dup_base * 100
        print(f"Снижение повторов: {reduction:.1f}%")
        if reduction >= 30:
            print("Условие выполнено. Снижение ≥30%")
        else:
            print("Условие не выполнено. Снижение <30%")
    else:
        print("В baseline нет повторов — ограничение не применимо")


if __name__ == '__main__':
    main()
