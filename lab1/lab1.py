import argparse
import re
from functools import lru_cache
from pathlib import Path

from pymorphy3 import MorphAnalyzer
from razdel import sentenize, tokenize


RUSSIAN_WORD = re.compile(r"[а-яё]+(?:-[а-яё]+)*", re.IGNORECASE)
CASE_MAP = {"gen2": "gent", "loc2": "loct", "acc2": "accs"}

# Проверка части речи
def is_target(parse):
    tag = parse.tag
    return tag.POS == "NOUN" or (tag.POS == "ADJF" and "Apro" not in tag) # Не берем местоименные прилагательные (apro)

# Берём лемму и часть речи из первого разбора, но сохраняем варианты числа и падежа
def preferred_parses(parses):
    first = parses[0]
    return [
        p for p in parses
        if p.normal_form == first.normal_form and p.tag.POS == first.tag.POS
        and is_target(p)
    ]

# Собирает запасные разборы
def alternative_parses(parses):
    first = parses[0]
    if is_target(first):
        return [p for p in parses if is_target(p)]
    if first.tag.POS in {"VERB", "INFN"}:
        return [p for p in parses if p.tag.POS == "NOUN" and "Surn" not in p.tag]
    return []

# Возвращает возможные роды слова
def genders(tag):
    if "ms-f" in tag:
        return {"masc", "femn"}
    return {tag.gender} if tag.gender else set()

# Сравнивает число, падеж и род двух разборов
def agree(left, right):
    left_case = CASE_MAP.get(left.case, left.case)
    right_case = CASE_MAP.get(right.case, right.case)
    if not left_case or left_case != right_case:
        return False
    if not left.number or left.number != right.number:
        return False

    left_genders, right_genders = genders(left), genders(right)
    if left.number == "plur" and (not left_genders or not right_genders):
        return True  # У прилагательных во множественном числе род не выражен
    return bool(left_genders & right_genders)


def match_pair(left, right):
    if not left or not right:
        return None

    right_preferred = preferred_parses(right)
    for a in preferred_parses(left):
        for b in right_preferred:
            if agree(a.tag, b.tag):
                return a.normal_form, b.normal_form

    # Прилагательное и существительное могут подсказать разбор друг друга,
    # даже если оба слова неоднозначны (например «мёртвые ели»)
    right_alternatives = alternative_parses(right)
    matches = (
        (a, b)
        for a in alternative_parses(left)
        for b in right_alternatives
        if {a.tag.POS, b.tag.POS} == {"ADJF", "NOUN"} and agree(a.tag, b.tag)
    )
    best = max(matches, key=lambda pair: pair[0].score * pair[1].score, default=None)
    if best:
        return best[0].normal_form, best[1].normal_form
    return None


def find_pairs(text):
    morph = MorphAnalyzer()
    parse_word = lru_cache(maxsize=8192)(morph.parse)

    for sentence in sentenize(text):
        words = [
            parse_word(token.text.lower()) if RUSSIAN_WORD.fullmatch(token.text) else []
            for token in tokenize(sentence.text)
        ]
        for left, right in zip(words, words[1:]):
            pair = match_pair(left, right)
            if pair:
                yield pair


def main():
    parser = argparse.ArgumentParser(description="Поиск согласованных соседних слов")
    parser.add_argument("file", type=Path, help="Путь к текстовому файлу в UTF-8")
    args = parser.parse_args()

    try:
        text = args.file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        parser.error(f"Не удалось прочитать файл {args.file}: {error}")

    for left, right in find_pairs(text):
        print(left, right)


if __name__ == "__main__":
    main()
