from transformers import pipeline
import argparse

def generate_text(promt, min_lenght, max_lenght):
    """
    Генерирует текст на основе промпта с использованием модели GPT-2.
    
    Использует Hugging Face Transformers pipeline для генерации текста с заданными
    параметрами длины. Функция обрабатывает возможные ошибки генерации.
    
    Parameters
    ----------
    prompt : str
        Входной текст (промпт), на основе которого генерируется продолжение
    min_length : int
        Минимальная длина генерируемого текста в токенах
    max_length : int
        Максимальная длина генерируемого текста в токенах
        
    Returns
    -------
    list or None
        Список с результатами генерации текста, где каждый элемент содержит:
        - 'generated_text': сгенерированный текст
        Возвращает None в случае возникновения ошибки
        
    Raises
    ------
    Exception
        Любые исключения, возникающие в процессе генерации текста, 
        перехватываются и выводятся в консоль
    """
    try:
        generator = pipeline(
            "text-generation",  
            model="gpt2"        
        )
        result = generator(
            promt,           
            min_length=min_lenght,    
            max_length=max_lenght,
            min_new_tokens=min_lenght,
            max_new_tokens=max_lenght     
        )
        return result
    
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None

def check_result_lenght(result, min_lenght, max_lenght):
    """
    Проверяет длину сгенерированного текста.
    
    Parameters
    ----------
    result : list
        Результат генерации текста
        
    Returns
    -------
    bool
        True если длина соответствует требованиям, False если нет
    """
    try:
        if result is None or len(result) == 0:
            print("Результат пустой")
            return False
            
        generated_text = result[0]["generated_text"]
        text_length = len(generated_text.split())
        
        print(f"Длина текста: {text_length} слов")
        
        if min_lenght <= text_length <= max_lenght:
            print(f"Длина соответствует требованиям ({min_lenght}-{max_lenght} слов)")
            return True
        else:
            print(f"Длина НЕ соответствует требованиям ({min_lenght}-{max_lenght} слов)")
            return False
            
    except Exception as e:
        print(f"Ошибка при проверке длины: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("min_lenght") 
    parser.add_argument("max_lenght")
    args = parser.parse_args()

    min_lenght = int(args.min_lenght)
    max_lenght = int(args.max_lenght)

    for i in range(1, 4):
        print(i, "итерация:\n")
        try:
            prompt = input("enter prompt: ")
            result = generate_text(prompt, min_lenght, max_lenght)

            if result is not None:
                print("answer", result[0]["generated_text"])
            else:
                print("Не удалось сгенерировать текст")
        except KeyboardInterrupt:
            print("\nПрогрмма прервана")
        except Exception as e:
            print(f"Ошибка: {e}")

        print("\nДлина в интервале?:", check_result_lenght(result, min_lenght, max_lenght))

if __name__ == '__main__':
    main()
