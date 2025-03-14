import os
from colorama import Fore, Style
from transformers import pipeline
import time
import warnings

# Suppress the FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)

def summarize_text(text):
    """Generate an abstractive summary of the text using a pre-trained model."""
    if not text.strip():
       return f"\n{Fore.RED}⛔️ No text provided."
   
    # Initialize the summarization pipeline (downloads model on first run)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Ensure text is within model constraints (truncate if too long)
    max_input_length = 1024 # BARTS max input length
    if len(text) > max_input_length:
       text = text[:max_input_length]

   # Generate summary
    try:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"\n{Fore.RED}⛔️ Error summarizing text: {e}"
    

def list_file_paths():
    # Get the directory of the current Python script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the files folder in the parent directory
    texts_dir = os.path.join(current_script_dir, 'texts')
    
    # Check if the texts directory exists
    if not os.path.exists(texts_dir):
        print(f"Texts directory not found: {texts_dir}")
        return []
    
    # List all files in the texts directory
    texts_files = os.listdir(texts_dir)
    
    # Get full paths for text files
    texts_paths = [os.path.join(texts_dir, img) for img in texts_files]
    
    # Print out the paths
    print(f"\n{Fore.GREEN}Text Paths:{Style.RESET_ALL}")
    for index, path in enumerate(texts_paths):
        print(f"{index + 1}: {os.path.basename(path)}")
    
    return texts_paths

if __name__ == "__main__":
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 6!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*      TEXT SUMMARIZER    *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*          📝🟰          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")

    available_texts = list_file_paths()

    selected_text = int(input(f"\n{Fore.YELLOW}Select a text file (1-{len(available_texts)}): {Style.RESET_ALL}"))

    try:
        with open(available_texts[selected_text-1], 'r', encoding='utf-8') as file:
            text = file.read()
            slow_print(f"\n{Fore.YELLOW}Summarizing Text: {selected_text}{Style.RESET_ALL}")
            slow_print(f"\n{Fore.CYAN}Original Text Preview (first 200 characters):{Style.RESET_ALL}")
            slow_print(text[:200] + "..." if len(text) > 200 else text)
            slow_print(f"\n{Fore.YELLOW}{'*' * 27}")
            slow_print(f"\n{Fore.YELLOW}Summarizing Text...{Style.RESET_ALL}")

            summary = summarize_text(text)
            slow_print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
            slow_print(summary)
            slow_print(f"\n{Fore.YELLOW}{'*' * 27}")

    except Exception as e:
        slow_print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    

