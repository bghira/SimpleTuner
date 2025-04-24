import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as compare_ssim
import imutils


def process_video(input_video_path, output_folder, detect_faces=False):
    cap = cv2.VideoCapture(input_video_path)

    frame_counter = 0
    previous_faces = {}
    last_detected_image = None

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            resized_frame = resize_image(frame)
            cropped_frame = crop_image(resized_frame)
            if detect_faces:
                faces_detected = detect_faces_in_image(cropped_frame)
                for x, y, w, h in faces_detected:
                    face_crop = cropped_frame[y : y + h, x : x + w]
                    face_key = f"{x}_{y}_{w}_{h}"
                    score = 0.0
                    if last_detected_image is not None:
                        score = image_difference(last_detected_image, cropped_frame)
                    if last_detected_image is None or score < 0.21:
                        cv2.imwrite(
                            os.path.join(
                                output_folder, f"frame_{frame_counter:05d}.jpg"
                            ),
                            cropped_frame,
                        )
                        previous_faces[face_key] = face_crop
                        last_detected_image = cropped_frame
            else:
                score = 0.0
                if last_detected_image is not None:
                    score = image_difference(last_detected_image, cropped_frame)
                if last_detected_image is None or score < 0.21:
                    cv2.imwrite(
                        os.path.join(output_folder, f"frame_{frame_counter:05d}.jpg"),
                        cropped_frame,
                    )
                    last_detected_image = cropped_frame
            frame_counter += 1

        else:
            break

    # Release everything after the job is finished
    cap.release()
    cv2.destroyAllWindows()


def resize_image(image, height=768):
    # resizing image while maintaining aspect ratio
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def crop_image(image):
    height, width = image.shape[:2]
    cropped_image = image[0:768, (width - 768) // 2 : (width + 768) // 2]
    return cropped_image


def detect_faces_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    faces_str = f"{faces}"
    if faces_str != "()":
        logging.info(f"Found faces {faces}")
    return faces


def image_difference(imageA, imageB):
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    logging.info(f"Returning score {score}")
    return score


from pathlib import Path


def process(input_path, output_folder, detect_faces=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isdir(input_path):
        # If input path is a directory, process each image file in the directory
        for image_file in Path(input_path).glob("*"):
            image = cv2.imread(str(image_file))
            process_image(image, output_folder, detect_faces)
    else:
        # If input path is not a directory, assume it's a video file and process it
        process_video(input_path, output_folder, detect_faces)


def process_image(image, output_folder, detect_faces=False):
    frame_counter = 0
    previous_faces = {}
    last_detected_image = None

    resized_image = resize_image(image)
    cropped_image = crop_image(resized_image)

    if detect_faces:
        faces_detected = detect_faces_in_image(cropped_image)
        for x, y, w, h in faces_detected:
            face_crop = cropped_image[y : y + h, x : x + w]
            face_key = f"{x}_{y}_{w}_{h}"
            score = 0.0
            if last_detected_image is not None:
                score = image_difference(last_detected_image, cropped_image)
            if last_detected_image is None or score < 0.21:
                cv2.imwrite(
                    os.path.join(output_folder, f"frame_{frame_counter:05d}.jpg"),
                    cropped_image,
                )
                previous_faces[face_key] = face_crop
                last_detected_image = cropped_image
    else:
        score = 0.0
        if last_detected_image is not None:
            score = image_difference(last_detected_image, cropped_image)
        if last_detected_image is None or score < 0.21:
            cv2.imwrite(
                os.path.join(output_folder, f"frame_{frame_counter:05d}.jpg"),
                cropped_image,
            )
            last_detected_image = cropped_image
    frame_counter += 1


if __name__ == "__main__":
    input_path = (
        "/notebooks/datasets/faces"  # path to the input video or image directory
    )
    output_folder = "/notebooks/datasets/processed_faces"  # path to the output folder
    process(input_path, output_folder, detect_faces=False)
