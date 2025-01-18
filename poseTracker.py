import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose detection
    result = pose.process(rgb_frame)

    # If pose landmarks are detected
    if result.pose_landmarks:
        # Draw landmarks on the frame
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Access the landmarks
        landmarks = result.pose_landmarks.landmark

        # Get the points for left shoulder, right shoulder, left hip, and right hip
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate the average point (belly button approximation)
        avg_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        avg_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        avg_z = (left_shoulder.z + right_shoulder.z + left_hip.z + right_hip.z) / 4

        # Convert normalized coordinates to pixel coordinates
        frame_height, frame_width, _ = frame.shape
        avg_x_pixels = int(avg_x * frame_width)
        avg_y_pixels = int(avg_y * frame_height)

        # Draw a circle at the calculated position
        cv2.circle(frame, (avg_x_pixels, avg_y_pixels), radius=5, color=(0, 255, 0), thickness=-1)

        # Display the coordinates

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
