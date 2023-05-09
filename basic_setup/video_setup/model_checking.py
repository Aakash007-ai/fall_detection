from tensorflow import keras

lstm_model = keras.models.load_model('../lstm_model.h5')
print(lstm_model)


import cv2
import mediapipe as mp
import numpy as np
import time
import math
from numpy import array
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'fall_demo.mp4'

cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter object.
save_name = f"{video_path.split('/')[-1].split('.')[0]}"
out = cv2.VideoWriter(f"{save_name}_keypoint.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while(cap.isOpened()):
        # Capture each frame of the video.
        ret, frame = cap.read()
        if ret:
            image = frame
            # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image = cv2.resize(image, (640, 480))
            
            # Get the start time.
            start_time = time.time()
            results = pose.process(image)
            # Get the end time.
            end_time = time.time()
            # Get the fps.
            fps = 1 / (end_time - start_time)
            # Add fps to total fps.
            total_fps += fps
            # Increment frame count.
            frame_count += 1    

            # Draw the pose on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks is not None:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                 # calculate the coordinates of relevant keypoints
                keypoints = []
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility)

                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y)
                keypoints.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].visibility)

                keypoints = array(keypoints)

                keypoints= keypoints.reshape((1, 1, len(keypoints)))
                res = lstm_model.predict(keypoints)

                if res>0.5:
                    cv2.putText(image, f"Person Fell down with probablity {res}", (11, 100), 0, 1, [0, 0, 255], thickness=3,lineType=cv2.LINE_AA)
                else:
                    cv2.putText(image, f"Person probablity {res}", (11, 100), 0, 1, [0, 255, 255], thickness=3,lineType=cv2.LINE_AA)


                # # check if the person fell down
                # len_factor = math.sqrt(
                #     ((left_shoulder.y - left_body.y) ** 2 + (left_shoulder.x - left_body.x) ** 2))
                # if left_shoulder.y > left_foot.y - len_factor and left_body.y > left_foot.y - (len_factor / 2) and left_shoulder.y > left_body.y - (len_factor / 2):
                #     # draw rectangle and text on the frame
                #     cv2.rectangle(image, (int(left_shoulder.x), int(left_shoulder.y)), (int(right_foot.x), int(right_foot.y)),
                #               color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                #     cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3,
                #             lineType=cv2.LINE_AA)
                
            # Write the FPS on the current frame.
            cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # Convert from BGR to RGB color format.
            cv2.imshow('image', image)
            out.write(image)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
# Calculate and print the average FPS.
# avg_fps = total_fps / frame_count
# print(f"Average FPS: {avg_fps:.3f}")
