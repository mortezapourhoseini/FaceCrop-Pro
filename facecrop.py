# Configuration
INPUT_DIR = "pre_dataset"           # Directory containing input images
OUTPUT_DIR = "new_dataset"         # Directory to save cropped face images
GRAYSCALE = False                     # Convert cropped faces to grayscale
PADDING = 0                         # Padding around face
CHECK_DUPLICATES = False              # Enable/disable duplicate face filtering
BLUR_THRESHOLD = 30.0               # Blur detection threshold
FACE_SIZE = (224, 224)               # Fixed output size of face crops

# App
import cv2
from pathlib import Path

input_path = Path(INPUT_DIR)
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
seen_faces = []  # For storing feature vectors of seen faces


def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_duplicate(face_crop, threshold=1000):
    resized = cv2.resize(face_crop, FACE_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = cv2.normalize(hist, hist).flatten()
    
    for existing in seen_faces:
        dist = cv2.norm(hist, existing, cv2.NORM_L2)
        if dist < threshold:
            return True
    seen_faces.append(hist)
    return False

def main():
    for image_file in input_path.glob("*.jpg"):
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"[WARN] Unable to read: {image_file}")
            continue

        if is_blurry(img):
            print(f"[INFO] Skipping blurry image: {image_file.name}")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"[INFO] No faces found in: {image_file.name}")
            continue

        for i, (x, y, w, h) in enumerate(faces):
            # Add padding and clamp
            x_pad = max(x - PADDING, 0)
            y_pad = max(y - PADDING, 0)
            x2_pad = min(x + w + PADDING, img.shape[1])
            y2_pad = min(y + h + PADDING, img.shape[0])
            face_crop = img[y_pad:y2_pad, x_pad:x2_pad]

            # Resize to fixed size
            face_crop = cv2.resize(face_crop, FACE_SIZE)

            if CHECK_DUPLICATES and is_duplicate(face_crop):
                print(f"[INFO] Skipping duplicate face in {image_file.name}")
                continue

            if GRAYSCALE:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            output_filename = output_path / f"{image_file.stem}_face{i}.jpg"
            cv2.imwrite(str(output_filename), face_crop)
            print(f"[INFO] Saved: {output_filename.name}")

    print("\n[INFO] Face extraction completed.")


if __name__ == "__main__":
    main()