#C:\Users\Dell\AppData\Local\Programs\Python\Python311\Scripts\
#C:\Python312\Scripts\
import os
import cv2
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import mediapipe as mp
import numpy as np

# Define face shape categories
face_shape_labels = ['Oval', 'Round', 'Square', 'Heart']

# Define a function to extract features from facial landmarks
def extract_features(face_landmarks):
    jawline = face_landmarks[0:17]
    nose_bridge = face_landmarks[27:31]
    left_eye = face_landmarks[36:42]
    right_eye = face_landmarks[42:48]

    # Calculate face width and height
    face_width = math.dist(jawline[0], jawline[-1])
    face_height = math.dist(nose_bridge[0], jawline[-1])

    # Calculate additional features
    eye_distance = math.dist(left_eye[0], right_eye[3])  # Distance between the outer corners of the eyes

    # Combine features into a feature vector
    feature_vector = [face_width, face_height, eye_distance]

    return feature_vector

# Preprocess an image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)

    # Convert back to RGB
    preprocessed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return preprocessed_image

# Load training data
X_train = []
y_train = []

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe FaceMesh with optimization parameters
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
) as face_mesh:

    train_data_dir = 'Dataset/training_set'

    for label in face_shape_labels:
        label_dir = os.path.join(train_data_dir, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = cv2.imread(image_path)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Resize the preprocessed image to a smaller size for faster processing
            smaller_image = cv2.resize(preprocessed_image, (640, 480))
            results = face_mesh.process(smaller_image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert face landmarks to a list of tuples
                    landmarks = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]
                    feature_vector = extract_features(landmarks)
                    X_train.append(feature_vector)
                    y_train.append(label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Create a dictionary to store example image file paths for each class
example_images_paths = {}
for label in face_shape_labels:
    example_images_paths[label] = []

# Populate the example image file paths
for label in face_shape_labels:
    label_dir = os.path.join(train_data_dir, label)
    example_image_paths = [os.path.join(label_dir, f'example_image{i}.jpg') for i in range(1, 4)]
    example_images_paths[label] = example_image_paths

# Define the range of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150, 200]  # You can add more values to search over
}

# Create the Random Forest classifier
clf = RandomForestClassifier(random_state=0)

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_n_estimators = grid_search.best_params_['n_estimators']

# Train the Random Forest classifier with the best hyperparameters
clf = RandomForestClassifier(n_estimators=best_n_estimators, random_state=0)
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate and print the classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy * 100}")

# Load your custom image
custom_image_path = 'image.jpg'
custom_image = cv2.imread(custom_image_path)

# Preprocess the custom image
preprocessed_custom_image = preprocess_image(custom_image)

# Initialize MediaPipe FaceMesh for the custom image
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
) as face_mesh:

    # Resize the preprocessed custom image to a smaller size for faster processing
    smaller_image = cv2.resize(preprocessed_custom_image, (640, 480))
    results = face_mesh.process(smaller_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert face landmarks to a list of tuples
            landmarks = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]
            feature_vector = extract_features(landmarks)

            # Predict face shape
            predicted_label = clf.predict([feature_vector])[0]

            # Display the original and predicted labels
            cv2.putText(smaller_image, f"Original: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Custom Image", smaller_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Display the best matches from the dataset
def find_best_match(custom_feature_vector, X_train, y_train):
    custom_feature_vector = np.array(custom_feature_vector).reshape(1, -1)
    distances = []

    for train_feature_vector in X_train:
        distance = np.linalg.norm(custom_feature_vector - train_feature_vector)
        distances.append(distance)

    best_match_indices = np.argsort(distances)[:3]
    best_matches = [(X_train[i], y_train[i]) for i in best_match_indices]
    return best_matches

# Display the best matches from the dataset
for i in range(3):
    best_matches = find_best_match(feature_vector, X_train, y_train)

    # Determine the fixed height for display
    fixed_height = 200

    # Create a list to store the images for display
    display_images = []

    for j, (best_match_feature_vector, best_match_label) in enumerate(best_matches):
        # Check if the best match is from the training dataset
        if best_match_label in y_train:
            # Adjust the file path based on your dataset naming convention
            best_match_image_name = f"{best_match_label} ({j+1}).jpg"
            best_match_image_path = os.path.join(train_data_dir, best_match_label, best_match_image_name)

            # Load the best match image
            best_match_image = cv2.imread(best_match_image_path)

            if best_match_image is not None:
                # Resize both the custom image and the best match image to the fixed height
                custom_image_resized = cv2.resize(preprocessed_custom_image, (int(fixed_height * preprocessed_custom_image.shape[1] / preprocessed_custom_image.shape[0]), fixed_height))
                best_match_image_resized = cv2.resize(best_match_image, (int(fixed_height * best_match_image.shape[1] / best_match_image.shape[0]), fixed_height))

                display_images.append(custom_image_resized)
                display_images.append(best_match_image_resized)

    # Display all images in a single row
    row_image = np.hstack(tuple(display_images))
    cv2.imshow(f"Custom Image vs. Best Matches", row_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
