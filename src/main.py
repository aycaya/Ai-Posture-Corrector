import cv2
import mediapipe as mp
import numpy as np

# --- 1. Helper Function: Calculate Angle (Trigonometry) ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points: a, b, and c.
    a: First point (Ear)
    b: Mid point (Shoulder)
    c: End point (Vertical Reference)
    """
    a = np.array(a) # Convert to numpy array
    b = np.array(b)
    c = np.array(c)
    
    # Calculate angle using arctan2 (handles x/y coordinates correctly)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure the angle is within 180 degrees
    if angle > 180.0:
        angle = 360 - angle
        
    return angle



def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize MediaPipe Drawing Utils (To draw lines on the screen easily)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize video capture from the default camera

    # 0 usually refers to the default camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    print("Press 'q' to quit.")
    
    # Read and display frames in a loop
    while True:
        # Capture frame-by-frame
        # Read a frame
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break  
        
        # Convert the BGR image to RGB before processing OpenCV uses BGR (Blue-Green-Red), but MediaPipe needs RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and find pose landmarks 'results' contains all the detected landmarks (coordinates)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # 1. Get Coordinates
            landmarks = results.pose_landmarks.landmark
            
            # Left Ear and Left Shoulder
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Vertical Reference (Upwards from shoulder)
            vertical_point = [shoulder[0], shoulder[1] - 0.5]

            # 2. Calculate Angle
            neck_angle = calculate_angle(ear, shoulder, vertical_point)

            # 3. Logic: Determine Color based on Threshold (35 degrees)
            if neck_angle > 35:
                status = "DURUS: BOZUK!"
                color = (0, 0, 255) # RED
            else:
                status = "DURUS: IYI"
                color = (0, 255, 0) # GREEN

            # 4. Visualization
            # Draw skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get pixel coordinates
            h, w, _ = frame.shape
            text_x = int(shoulder[0] * w)
            text_y = int(shoulder[1] * h)

            # Draw the Angle Value
            cv2.putText(frame, f"Angle: {int(neck_angle)}", (text_x + 10, text_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Draw the Status Text (Big Warning)
            cv2.putText(frame, status, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            
            # Draw a circle on the shoulder pivot
            cv2.circle(frame, (text_x, text_y), 7, color, -1)

        # Display the resulting frame
        cv2.imshow("Posture Corrector - Skeleton View", frame)
        #waitkey function to wait for a key event for 1 ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()