import cv2
import os
import time
from colorama import init, Fore, Style

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

def  detect_faces_on_webcam(face_cascade):
    """Detect faces in real-time on the webcam feed."""
    # Open webcam (0 is usually te default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception(f"{Fore.RED}\n⚠️ Webcam failed to open. Check if it's connected or try a different index (e.g., 1).{Style.RESET_ALL}")
    
    # Set webcam properties to stabilize feed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    slow_print(f"{Fore.GREEN}\n🎉 Webcam opened successfully. Press 'q' to quit the webcam feed.{Style.RESET_ALL}")

    max_retries = 3
    retry_count = 0

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                slow_print(f"{Fore.RED}\n⚠️ Failed to read frame from the webcam. Retrying ({retry_count}/{max_retries}).{Style.RESET_ALL}")
                time.sleep(0.1) # Brief delay before retry 
                if retry_count >= max_retries:
                    slow_print(f"{Fore.RED}\n⚠️ Failed to read frame after {max_retries} retries. Exiting.{Style.RESET_ALL}")
                    break
                continue

            # Reset retry count on successful frame capture
            retry_count = 0

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30)
            )

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Webcam Face Detector", frame)

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                slow_print("\n🚫 Exiting the webcam feed.")
                break
        
    finally: 
       # Release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        slow_print(f"{Fore.MAGENTA}\n📵 Webcam feed closed.{Style.RESET_ALL}")

if __name__ == "__main__":
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 10!   *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}*   WEBCAM FACE DETECTOR  *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*          😁🕵🏻📹          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")

    try:
        # Load the face cascade classifier
        face_cascade = load_face_cascade()
        detect_faces_on_webcam(face_cascade)
        
    except Exception as e:
        slow_print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    finally:
        # Final cleanup to ensure all windows close
        cv2.destroyAllWindows()