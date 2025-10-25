import asyncio
import base64
import pickle

import cv2
import face_recognition
import numpy as np
from frame_sdk import Frame
from frame_sdk.display import Alignment

# Load the known faces and embeddings
encodings_path = "encodings.pickle"
with open(encodings_path, "rb") as f:
    data = pickle.load(f)
print("Encodings loaded.")


async def main():
    try:
        # Initialize the Frame device
        async with Frame() as frame:
            print("Connected to Frame.")
            print("Starting video stream. Press 'q' to exit.")

            last_displayed_name = None

            while True:
                # Capture an image from the Frame camera
                image = await frame.camera.take_photo()
                if image is None:
                    print("Failed to capture image.")
                    continue

                # Process the image based on its type
                if isinstance(image, bytes):
                    # Image is a byte string (e.g., JPEG data)
                    image_data = np.frombuffer(image, np.uint8)
                    frame_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                elif isinstance(image, str):
                    # Image is a base64-encoded string
                    image_bytes = base64.b64decode(image)
                    image_data = np.frombuffer(image_bytes, np.uint8)
                    frame_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                elif "PIL.Image" in str(type(image)):
                    # Image is a PIL Image
                    frame_image = np.array(image)
                    frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                else:
                    print("Unsupported image type.")
                    continue

                # Resize frame for faster processing
                small_frame = cv2.resize(frame_image, (0, 0), fx=0.25, fy=0.25)

                # Convert BGR to RGB (face_recognition uses RGB)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all face locations and face encodings in the frame
                face_locations = face_recognition.face_locations(
                    rgb_small_frame, model="hog"
                )
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )

                face_names = []

                for face_encoding in face_encodings:
                    # Compare face encodings with known faces
                    matches = face_recognition.compare_faces(
                        data["encodings"], face_encoding, tolerance=0.6
                    )
                    name = "Unknown"

                    # Use the known face with the smallest distance
                    face_distances = face_recognition.face_distance(
                        data["encodings"], face_encoding
                    )
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = data["names"][best_match_index]

                    face_names.append(name)

                # Determine the name to display
                if not face_names:
                    name_to_display = "No face detected"
                else:
                    name_to_display = face_names[0]  # Display the first recognized face

                # Update the Frame display if the name has changed
                if name_to_display != last_displayed_name:
                    # Display the recognized name on the Frame device
                    await frame.display.show_text(
                        f"{name_to_display}", align=Alignment.MIDDLE_CENTER
                    )
                    print(f"Displayed on Frame: {name_to_display}")
                    last_displayed_name = name_to_display

                # Optional: Display the results in a window
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a rectangle around the face
                    cv2.rectangle(
                        frame_image, (left, top), (right, bottom), (0, 255, 0), 2
                    )
                    # Draw a label below the face
                    cv2.rectangle(
                        frame_image,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    # Write the name of the person
                    cv2.putText(
                        frame_image,
                        name,
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        1,
                    )

                # Show the video frame with the bounding boxes and names (optional)
                cv2.imshow("Video", frame_image)

                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Add a small delay to prevent high CPU usage
                await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())