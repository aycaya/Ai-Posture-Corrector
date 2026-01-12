import cv2
import mediapipe as mp

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
            # Draw the dots and connections (skeleton) on the original 'frame'
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
           

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