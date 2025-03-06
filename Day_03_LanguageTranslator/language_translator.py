from colorama import init, Fore, Style
from deep_translator import GoogleTranslator
import time

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)

def print_supported_languages():
    """Prints a list of supported languages."""
    langs = GoogleTranslator().get_supported_languages(as_dict=True)
    slow_print("\n🔠 Supported Languages:")
    count = 1
    for lang, abbr in langs.items():
        print(f"{count}: {lang} ({abbr})")
        count+=1
    slow_print("\n🤖💬")
    slow_print(f"{Fore.YELLOW} ⬆️ === Above are the supported languages === ⬆️ {Style.RESET_ALL}")

def translate_text(text, target_lang):
    """Translates text to the specified language."""
    try: 
        translator = GoogleTranslator(source="en", target=target_lang)
        tranlsation = translator.translate(text)
        return tranlsation
    except Exception as e:
        return f"Error {e}"
    
def get_user_target_language():
    slow_print(f"{Fore.GREEN} Staring with English, which language would you like to translate to? {Style.RESET_ALL}")
    selected_lang = input(f"{Fore.MAGENTA} Enter language code for desired translation (e.g., 'en' for English, will default to English): {Style.RESET_ALL}")
    return selected_lang

def translate_user_input(selected_lang):
    text_to_translate = input(f"\n{Fore.BLUE} Enter the text you want to translate: {Style.RESET_ALL}")
    translation = translate_text(text_to_translate, selected_lang)
    slow_print(f"\n🌐 Translation")
    slow_print(f"{Fore.GREEN}{selected_lang}: {translation} {Style.RESET_ALL}")


def main():
    # Arcade-style welcome message with color and emojis
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 3!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*   LANGUAGE TRANSLATOR   *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*   🗣️  TRANSLATE IT!  🗣️   *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")

    continue_program = True

    print_supported_languages()

    # Get target language
    selected_lang = get_user_target_language()
   
    # Get text to translate
    translate_user_input(selected_lang)

    while continue_program:
        # Present users options
        slow_print(f"\n{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
        slow_print("\n1: Translate again")
        slow_print("2: See Available Languages")
        slow_print("3: Change Target Language")
        slow_print("4: Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            translate_user_input(selected_lang)
        elif choice == "2":
            print_supported_languages()
        elif choice == "3":
            selected_lang = get_user_target_language()
            slow_print(f"\n{Fore.CYAN}Your selected language is: {selected_lang}{Style.RESET_ALL}")
        elif choice == "4":
            slow_print(f"\n{Fore.RED}Goodbye!")
            continue_program = False
       

       

if __name__ == "__main__":
    main()