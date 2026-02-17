import cv2
import mediapipe as mp
import numpy as np

# --- 1. Math Function: Calculate Angle ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    """
    a = np.array(a) # Ear
    b = np.array(b) # Shoulder
    c = np.array(c) # Vertical Reference
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- 2. Main Application Loop ---
def main():
    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Setup Camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # --- Calibration Variables ---
    target_angle = None 
    target_shoulder_y = None # New: To track if user slides down
    is_calibrated = False

    print("Program Started. Press 'c' to Calibrate, 'r' to Reset, 'q' to Quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        h, w, _ = frame.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- A. Get Coordinates ---
            # Ear (Target)
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            
            # Shoulder (Pivot & Height Reference)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            # Vertical Reference
            vertical_point = [shoulder[0], shoulder[1] - 0.5]

            # --- B. Calculate Current Angle ---
            current_angle = calculate_angle(ear, shoulder, vertical_point)
            
            # Get current shoulder height (Y coordinate)
            current_shoulder_y = shoulder[1]

            # --- C. Logic: Calibration & Posture Check ---
            
            if not is_calibrated:
                status = "Press 'c' to Calibrate"
                color = (255, 255, 0) # Cyan
                cv2.putText(frame, f"Current: {int(current_angle)}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            else:
                # --- LOGIC UPDATE: CHECK BOTH ANGLE AND HEIGHT ---
                
                # 1. Calculate Differences
                angle_diff = current_angle - target_angle
                
                # Height diff: If positive, it means shoulder went DOWN (in image coords, Y increases downwards)
                height_diff = current_shoulder_y - target_shoulder_y 
                
                # Threshold for sliding down (0.03 means 3% of screen height)
                SLUMP_THRESHOLD = 0.03 

                # 2. Determine Status
                
                # Priority 1: Check if sliding down/slumping
                if height_diff > SLUMP_THRESHOLD:
                    status = "ALERT: SLUMPING DOWN!"
                    color = (0, 0, 255) # Red
                
                # Priority 2: Check Neck Angle (User's custom strict values)
                elif angle_diff < 2:  
                    status = "POSTURE: GOOD"
                    color = (0, 255, 0) # Green
                elif angle_diff < 8: 
                    status = "WARNING: SLIGHT TILT"
                    color = (0, 255, 255) # Yellow
                else: 
                    status = "ALERT: NECK BENT!"
                    color = (0, 0, 255) # Red

                # Display Info
                info_text = f"Target: {int(target_angle)} | Current: {int(current_angle)} | Slide: {height_diff:.3f}"
                cv2.putText(frame, info_text, (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- D. Visualization ---
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            text_x = int(shoulder[0] * w)
            text_y = int(shoulder[1] * h)

            # Draw the status text
            cv2.putText(frame, status, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            
            # Draw shoulder point
            cv2.circle(frame, (text_x, text_y), 8, color, -1)
            
            # Draw a horizontal line to show the calibrated shoulder level (if calibrated)
            if is_calibrated:
                target_y_pixel = int(target_shoulder_y * h)
                cv2.line(frame, (0, target_y_pixel), (w, target_y_pixel), (255, 255, 255), 1)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and results.pose_landmarks:
            # Capture both Angle AND Shoulder Height
            target_angle = current_angle
            target_shoulder_y = shoulder[1]
            is_calibrated = True
            print(f"Calibration Set! Angle: {target_angle}, Height: {target_shoulder_y}")
        elif key == ord('r'):
            is_calibrated = False
            target_angle = None
            target_shoulder_y = None
            print("Calibration Reset.")

        cv2.imshow("AI Posture Corrector - Anti-Slump Mode", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()