import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,  # Allow detection of multiple faces
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

# Variables for tracking and click refresh
tracking_face_index = None
refresh_requested = False
click_x, click_y = None, None
refresh_radius = 100  # Pixels to define the search region
face_labels = {}  # Dictionary to map face indices to labels

# Function to handle mouse click event
def refresh_focus(event, x, y, flags, param):
    global refresh_requested, click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")
        refresh_requested = True
        click_x, click_y = x, y

# Set the mouse callback function
cv2.namedWindow('MediaPipe Face Mesh')
cv2.setMouseCallback('MediaPipe Face Mesh', refresh_focus)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect faces
    result = face_mesh.process(rgb_frame)

    # On refresh request, find faces near the clicked position
    if refresh_requested and result.multi_face_landmarks:
        refresh_requested = False  # Reset the flag
        closest_face_index = None
        min_distance = float('inf')

        for idx, face_landmarks in enumerate(result.multi_face_landmarks):
            # Calculate the average position of the face
            avg_x = sum([landmark.x for landmark in face_landmarks.landmark]) / len(face_landmarks.landmark)
            avg_y = sum([landmark.y for landmark in face_landmarks.landmark]) / len(face_landmarks.landmark)

            # Convert to pixel coordinates
            frame_height, frame_width, _ = frame.shape
            avg_x_pixels = int(avg_x * frame_width)
            avg_y_pixels = int(avg_y * frame_height)

            # Calculate distance between the clicked point and the face center
            distance = ((avg_x_pixels - click_x) ** 2 + (avg_y_pixels - click_y) ** 2) ** 0.5

            # Check if the face is within the refresh radius
            if distance < refresh_radius and distance < min_distance:
                min_distance = distance
                closest_face_index = idx

        if closest_face_index is not None: #it finds the closest face around the clicked area
            tracking_face_index = closest_face_index
            print("New face selected.", tracking_face_index)
        else:
            print("No faces clicked")

    # Draw landmarks and labels for all detected faces
    if result.multi_face_landmarks:
        for idx, face_landmarks in enumerate(result.multi_face_landmarks):
            # Assign or retrieve a label for each face
            if idx not in face_labels:
                face_labels[idx] = f"Face {len(face_labels) + 1}"

            # Get the label for the current face
            label = face_labels[idx]

            # Draw landmarks
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Calculate the average position of the face for labeling
            avg_x = sum([landmark.x for landmark in face_landmarks.landmark]) / len(face_landmarks.landmark)
            avg_y = sum([landmark.y for landmark in face_landmarks.landmark]) / len(face_landmarks.landmark)
            frame_height, frame_width, _ = frame.shape
            avg_x_pixels = int(avg_x * frame_width)
            avg_y_pixels = int(avg_y * frame_height)

            # Add a label near the face
            if idx == tracking_face_index:
                label = "Tracked Face"
                color = (0, 0, 255)  # Red for the tracked face
            else:
                color = (255, 255, 255)  # White for other faces

            cv2.putText(frame, label, (avg_x_pixels, avg_y_pixels - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame
    cv2.imshow('MediaPipe Face Mesh', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
