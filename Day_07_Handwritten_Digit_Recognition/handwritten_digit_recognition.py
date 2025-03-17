from colorama import Fore, Style
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import joblib
import os
import time

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)

def train_model():
    """Train an SVC model on the MNIST dataset."""
    # Check if model is cached
    model_file = "mnist_svc_model.pkl"
    if os.path.exists(model_file):
        print("Loading cached model...")
        start_time = time.time()
        model = joblib.load(model_file)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
        return model

    # Load MNIST (takes a minute the first time)
    slow_print(f"\n{Fore.LIGHTGREEN_EX}Fetching MNIST dataset (may take a minute)...{Style.RESET_ALL}")
    start_time = time.time()
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
        slow_print(f"\n{Fore.MAGENTA}Fetched MNIST dataset in {time.time() - start_time:.2f} seconds.{Style.RESET_ALL}")
        start_time = time.time()

        X = X / 255.0 * 16  # Scale to 0-16 like Digits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        slow_print(f"\n{Fore.LIGHTBLUE_EX}Training model...{Style.RESET_ALL}")
        model = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42, probability=True)  # Enable probabilities)
        model.fit(X_train, y_train)

        joblib.dump(model, model_file)
        slow_print(f"\n{Fore.MAGENTA}Model trained & saved in {time.time() - start_time:.2f} seconds.{Style.RESET_ALL}")
        return model
    except Exception as e:
        slow_print(f"\n{Fore.RED}Error training model: {e}{Style.RESET_ALL}")
        return None

def classify_user_digit(model):
    """Load 'handwritten-digits/digit-1.png', classify it, and show original vs processed with prediction."""
    if model is None:
        print("Model not trained successfully. Exiting.")
        return
    
    file_path = "handwritten-digits/digit-1.png"
    
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please add 'digit-1.png' to 'handwritten-digits' folder.")
        return
    
    try:
        slow_print(f"\n{Fore.GREEN}Processing image...{Style.RESET_ALL}")
        start_time = time.time()

        # Load original image
        original_img = Image.open(file_path).convert("L")
        original_array = np.array(original_img, dtype=np.float64)
        original_array = np.where(original_array > 100, 255, 0)
        original_img = Image.fromarray(original_array.astype(np.uint8))
        
        # Process to 28x28 (MNIST size)
        processed_img = original_img.resize((28, 28), Image.Resampling.LANCZOS)
        digit_array = np.array(processed_img, dtype=np.float64) / 255.0 * 16
        digit_array = 16 - digit_array.flatten()  # Invert
        
        # Predict
        slow_print(f"\n{Fore.YELLOW}Predicting...{Style.RESET_ALL}")
        prediction = model.predict([digit_array])[0]
        slow_print(f"\n{Fore.BLUE}Prediction completed in {time.time() - start_time:.2f} seconds.{Style.RESET_ALL}")
        
        # Display original and processed
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ax1.imshow(original_array, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        ax2.imshow(digit_array.reshape(28, 28), cmap='gray')
        ax2.set_title(f"Processed (28x28)\nPredicted: {prediction}")
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")

if __name__ == "__main__":
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 7!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*   DIGIT RECOGNITION    *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*           🔢✅          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")

    model = train_model()
    classify_user_digit(model)