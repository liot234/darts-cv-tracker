import cv2
import math
import time
import numpy as np
import imagediff  # My custom module

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
CAM_0_INDEX = 0 # TOP CAM
CAM_1_INDEX = 1 #SIDE CAM

# Board callibration constants 
ROI_Y_START = 170
ROI_Y_END = 290
PERFECT_CENTER_X = 254
PERFECT_CENTER_Y = 246


COOLDOWN_SECONDS = 3.0

# Dartboard segments listed clockwise 
BOARD_SEGMENTS = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

# Homography Calibration Points
DISTORTED_POINTS = np.array([[130, 331], [368, 628], [628, 338], [388, 9]], dtype='float32')
PERFECT_SQUARE_POINTS = np.array([
    [450, 250],   
    [250, 450],  
    [50, 250],   
    [250, 50]    
], dtype='float32')

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_homography_matrix():
    #Defines the matrix map that undistorts my grid 
    matrix, _ = cv2.findHomography(DISTORTED_POINTS, PERFECT_SQUARE_POINTS)
    return matrix

def reset_backgrounds(cam1, cam2):
    
    print("Round over! Wiping memory and grabbing fresh backgrounds...")
    time.sleep(5)
    
    # Flush buffers
    for _ in range(50):
        cam1.read()
        cam2.read()
        
    ret1, base1 = cam1.read()
    ret2, base2 = cam2.read()
    
    if ret1 and ret2:
        roi_base1 = base1[ROI_Y_START:ROI_Y_END, :]
        roi_base2 = base2[ROI_Y_START:ROI_Y_END, :]
        print("Board reset. Ready for next throw!")
        return roi_base1, roi_base2
    
    raise RuntimeError("Failed to capture background frames during reset.")

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
def main():
    cam1 = cv2.VideoCapture(CAM_0_INDEX)
    cam2 = cv2.VideoCapture(CAM_1_INDEX)
    
    matrix = get_homography_matrix()
    
    # Warming up that camera 
    for _ in range(100):
        cam1.read()
        cam2.read()
        
    ret1, base1 = cam1.read()
    ret2, base2 = cam2.read()
    
    if not (ret1 and ret2):
        print("Error: Could not initialize cameras.")
        return

    roi_base1 = base1[ROI_Y_START:ROI_Y_END, :]
    roi_base2 = base2[ROI_Y_START:ROI_Y_END, :]
    
    cam1_saved_x = []
    cam2_saved_x = []
    round_throws = []
    
    previous_time = time.time()
    delta = 0

    while True:
        current_time = time.time()
        delta += current_time - previous_time
        previous_time = current_time
        
        ret1, img1 = cam1.read()
        ret2, img2 = cam2.read()
        
        # Wait for cooldown period before processing next throw
        if delta > COOLDOWN_SECONDS and ret1 and ret2:
            delta = 0  # Reset timer
            
            roi_img1 = img1[ROI_Y_START:ROI_Y_END, :]
            roi_img2 = img2[ROI_Y_START:ROI_Y_END, :]
            
            # --- Image Processing ---
            masked1, cont1 = imagediff.im_diff(roi_img1, roi_base1, 35, True)
            roi_base1 = roi_img1.copy()
            
            masked2, cont2 = imagediff.im_diff(roi_img2, roi_base2, 40, False)
            roi_base2 = roi_img2.copy()
            
            bot1_point = None
            bot2_point = None

            # Process Camera 1
            if cont1 is not None:
                bot1_point = tuple(cont1[cont1[:, :, 1].argmax()][0])
                cv2.circle(roi_img1, bot1_point, 5, (0, 0, 255), -1)
                cv2.drawContours(roi_img1, cont1, -1, (0, 255, 0), 3)
                cv2.imshow('Cam 1 Mask', masked1)
                cv2.imshow('Cam 1 ROI', img1[ROI_Y_START:ROI_Y_END, :])

            # Process Camera 2
            if cont2 is not None:
                bot2_point = tuple(cont2[cont2[:, :, 1].argmax()][0])
                cv2.circle(roi_img2, bot2_point, 5, (0, 0, 255), -1)
                cv2.drawContours(roi_img2, cont2, -1, (0, 255, 0), 3)
                cv2.imshow('Cam 2 Mask', masked2)
                cv2.imshow('Cam 2 ROI', img2[ROI_Y_START:ROI_Y_END, :])

            # --- Occlusion Handling, if two darts are in the same line verically or horizontally ---
            if cont2 is not None and cv2.contourArea(cont2) < 10 and cam2_saved_x:
                closest_x = min(cam2_saved_x, key=lambda x: abs(x - bot2_point[0]))
                bot2_point = (closest_x, bot2_point[1])
                
            if cont1 is not None and cv2.contourArea(cont1) < 10 and cam1_saved_x:
                closest_x = min(cam1_saved_x, key=lambda x: abs(x - bot1_point[0]))
                bot1_point = (closest_x, bot1_point[1])

            # --- Point Calculation & Scoring ---
            if bot1_point and bot2_point:
                cam1_saved_x.append(bot1_point[0])
                cam2_saved_x.append(bot2_point[0])
                
                raw_points = np.array([[[bot1_point[0], bot2_point[0]]]], dtype='float32')
                perfect_points = cv2.perspectiveTransform(raw_points, matrix)
                
                flat_x = perfect_points[0][0][0]
                flat_y = perfect_points[0][0][1]
                round_throws.append((flat_x, flat_y))
                
                # Calculating Clustering, i.e. accuracy 
                if len(round_throws) > 1:
                    throws_arr = np.array(round_throws)
                    centroid = np.mean(throws_arr, axis=0)
                    distances = np.linalg.norm(throws_arr - centroid, axis=1)
                    mre_clustering = np.mean(distances)
                    print(f"Mean Radial Error (Clustering): {mre_clustering:.2f}")

                # Trigonometry for Segment Scoring
                dx = flat_x - PERFECT_CENTER_X
                dy = PERFECT_CENTER_Y - flat_y
                
                radius = math.sqrt(dx**2 + dy**2)
                angle_deg = math.degrees(math.atan2(dy, dx))
                shifted_angle = (angle_deg + 9 + 360) % 360  # Adjusted +9 for half-segment shift
                segment_index = int(shifted_angle / 18)
                
                hit_value = BOARD_SEGMENTS[segment_index]
                
                print(f"Radius: {radius:.2f}")
                print(f"Angle: {angle_deg:.2f}° | You hit the {hit_value}!")

       
        key = cv2.waitKey(1) & 0xFF
        #Quitting
        if key == ord('q'):
            break
        #For when you want to take darts out
        elif key == ord('b'):
            roi_base1, roi_base2 = reset_backgrounds(cam1, cam2)
            cam1_saved_x.clear()
            cam2_saved_x.clear()
            round_throws.clear()
            delta = 0

    # Clean up resources
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
    
