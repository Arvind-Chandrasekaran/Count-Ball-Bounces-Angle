import cv2
import numpy as np

H_LOW, S_LOW, V_LOW = 0, 0, 200
H_HIGH, S_HIGH, V_HIGH = 180, 30, 255

ball_color_lower = np.array([H_LOW, S_LOW, V_LOW], dtype=np.uint8)
ball_color_upper = np.array([H_HIGH, S_HIGH, V_HIGH], dtype=np.uint8)

background_color_lower = np.array([H_LOW, S_LOW, V_LOW], dtype=np.uint8)
background_color_upper = np.array([H_HIGH, S_HIGH, V_HIGH], dtype=np.uint8)

cap = cv2.VideoCapture('C:/ball_bouncing_white_2.mp4')

previous_positions = []
previous_angles = []
deviation_count = 0

num_frames_for_average = 10  # Choose the number of frames to average

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ball_mask = cv2.inRange(hsv_frame, ball_color_lower, ball_color_upper)
    background_mask = cv2.inRange(hsv_frame, background_color_lower, background_color_upper)

    combined_mask = cv2.bitwise_and(ball_mask, background_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        ball_contour = max(contours, key=cv2.contourArea)
        ball_position = tuple(ball_contour[0][0])

        previous_positions.append(ball_position)

        if len(previous_positions) > num_frames_for_average:
            previous_positions.pop(0)  # Remove the oldest position

        if len(previous_positions) == num_frames_for_average:
            average_position = np.mean(np.array(previous_positions), axis=0)

            current_angle = np.arctan2(ball_position[1] - average_position[1], ball_position[0] - average_position[0])

            previous_angles.append(current_angle)

            if len(previous_angles) > num_frames_for_average:
                previous_angles.pop(0)  # Remove the oldest angle

            if len(previous_angles) == num_frames_for_average:
                average_angle = np.mean(previous_angles)

                angle_deviation = np.abs(current_angle - average_angle)

                if angle_deviation > np.radians(185):
                    deviation_count += 1
                    print(f"Deviation {deviation_count}")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Total deviations: {deviation_count}")

cap.release()
cv2.destroyAllWindows()
