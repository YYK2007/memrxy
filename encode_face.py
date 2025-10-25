import os
import pickle

import cv2
import face_recognition

# Paths
dataset_path = "dataset" 
encodings_path = "encodings.pickle"

known_encodings = []
known_names = []

# Loop over the person folders
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue 

    # Loop over the image files for each person
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        print(f"Processing {image_path}...")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            continue

        # Convert BGR to RGB 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        boxes = face_recognition.face_locations(rgb_image, model="hog")
        if len(boxes) == 0:
            print(f"No face found in {image_path}")
            continue

        # Compute the facial embeddings
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        # Add each encoding and name to our set of known faces
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the facial encodings and names to disk
print("Serializing encodings...")
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print(f"Encodings saved to {encodings_path}")