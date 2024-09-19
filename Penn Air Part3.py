import numpy as np
import cv2 as cv

# Path to image
path = '/Users/bhsieh/Desktop/Penn Air/PennAir 2024 App Dynamic Hard.mp4'
output_path = '/Users/bhsieh/Desktop/Penn Air/Part 3 Processed Video.mp4'

# Open the video file or capture from a webcam
video_capture = cv.VideoCapture(path)

# Check if video opened successfully
if not video_capture.isOpened():
    print("Error opening video file")

# Get properties of the input video
fps = video_capture.get(cv.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the video
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
video_writer = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    
    if ret:
        average_color_per_channel = np.mean(frame, axis=(0, 1))
        average_color = np.round(average_color_per_channel).astype(int) - 25

        result = frame.astype(int) - average_color
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Test to show processed image
        cv.imshow('Clipped Video', result)

        lower_bound = np.array([0, 0, 0], dtype=np.uint8)
        upper_bound = np.array([40, 40, 40], dtype=np.uint8)
        mask = cv.inRange(result, lower_bound, upper_bound)

        result[mask > 0] = [0, 0, 0]

        # Test to show processed image
        cv.imshow('Masked Video', result)
        
        black_pixels = np.where(
            (result[:, :, 0] != 0) | 
            (result[:, :, 1] != 0) | 
            (result[:, :, 2] != 0)
        )

        result[black_pixels] = [255, 255, 255]

        # Test to show processed image
        cv.imshow('White Shapes Video', result)

        blur = cv.bilateralFilter(result,9,75,75)
        
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        fixed_contours = [contour for contour in contours if cv.contourArea(contour) > 3000]


        for contour in fixed_contours:
            M = cv.moments(contour)
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Draw a small circle (dot) at the centroid
            cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # Red dot with a radius of 5

            # Display the coordinates of the center point on the frame
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3
            text_color = (225, 225, 225)
            
            cv.putText(frame, f"coords: [{cX}, {cY}]", (cX - 50, cY - 20), font, font_scale, text_color, font_thickness)

        # Draw the contours on the original frame
        cv.drawContours(frame, fixed_contours, -1, (43, 75, 238), 6)

        # Save the processed frame to the video
        video_writer.write(frame)
        
        # Display the resulting frame
        cv.imshow('Final Video', frame)
        
        # Press 'q' to exit the video early
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close display windows
video_capture.release()
cv.destroyAllWindows()