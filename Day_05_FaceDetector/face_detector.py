import cv2
import time
from colorama import init, Fore, Style
import os

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)
def load_face_cascade():
    """Load the pre-trained face detection model."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if face_cascade.empty():
        raise Exception("⚠️ Face cascade file not found or unable to load.")
    
    return face_cascade

def detect_faces_in_image(image_path, face_cascade):
    """Detect faces in an image."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("⚠️ Image not found or unable to load.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30)
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Display the result
    cv2.imshow("Detected Faces", image)

    # Wait for a key press to close
    slow_print(f"\n{Fore.RED}Click inside image window and press any key to close...{Style.RESET_ALL}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def list_image_paths():
    # Get the directory of the current Python script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the images folder in the parent directory
    images_dir = os.path.join(current_script_dir, 'images')
    
    # Check if the images directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return []
    
    # List all files in the images directory
    image_files = os.listdir(images_dir)
    
    # Get full paths for image files
    image_paths = [os.path.join(images_dir, img) for img in image_files]
    
    # Print out the paths
    print(f"\n{Fore.GREEN}Image Paths:{Style.RESET_ALL}")
    for index, path in enumerate(image_paths):
        print(f"{index + 1}: {os.path.basename(path)}")
    
    return image_paths


if __name__ == "__main__":
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 5!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*     FACE DETECTOR    *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*          😁🕵🏻‍♂️          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")

    images = list_image_paths()
    selected_image = int(input(f"\n{Fore.YELLOW}Select an image (1-{len(images)}): {Style.RESET_ALL}"))

    try:
        # Load the face cascade classifier
        face_cascade = load_face_cascade()
        image_path = ""
        detect_faces_in_image(images[selected_image-1], face_cascade)

    except Exception as e:
        slow_print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    finally:
        # Final cleanup to ensure all windows close
        cv2.destroyAllWindows()