from transformers import pipeline
import argparse

def generate_text(prompt, min_length, max_length):
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
            prompt,           
            min_length=min_length,    
            max_length=max_length,
            min_new_tokens=min_length,
            max_new_tokens=max_length     
        )
        return result
    
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None

def check_result_length(result, min_length, max_length):
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
        
        if min_length <= text_length <= max_length:
            print(f"Длина соответствует требованиям ({min_length}-{max_length} слов)")
            return True
        else:
            print(f"Длина НЕ соответствует требованиям ({min_length}-{max_length} слов)")
            return False
            
    except Exception as e:
        print(f"Ошибка при проверке длины: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("min_length") 
    parser.add_argument("max_length")
    args = parser.parse_args()

    min_length = int(args.min_length)
    max_length = int(args.max_length)

    for i in range(1, 4):
        print(i, "итерация:\n")
        try:
            prompt = input("enter prompt: ")
            result = generate_text(prompt, min_length, max_length)

            if result is not None:
                print("answer", result[0]["generated_text"])
            else:
                print("Не удалось сгенерировать текст")
        except KeyboardInterrupt:
            print("\nПрограмма прервана")
        except Exception as e:
            print(f"Ошибка: {e}")

        print("\nДлина в интервале?:", check_result_length(result, min_length, max_length))

if __name__ == '__main__':
    main()
