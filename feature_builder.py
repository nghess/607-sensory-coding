import cv2
import numpy as np
from PIL import Image

def rotate_point(origin, point, angle):

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    return int(qx), int(qy)


# Generates feature to animate
def generate_feature(win, feature, angle=0, thickness=1):

    # Check that feature is correctly specified
    assert feature in ("cross", "tee", "elbow", "end"), "Feature must be one of 'cross', 'tee', 'elbow', or 'end'."

    # Convert angle to radians
    angle = np.deg2rad(angle)
    
    # Create feature_output matrix, if end feature is selected, generate end-detector mask
    feature_output = np.zeros((win, win), np.uint8)
    if feature == "end":
        end_mask = np.zeros((win, win), np.uint8)

    # Define center of feature_output
    center = win // 2
    origin = (center, center)

    # Draw Top
    if feature in ("cross", "tee", "end", "elbow"):
        endpoint = rotate_point(origin, (center, 1), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    # Draw Bottom
    if feature == "cross":
        endpoint = rotate_point(origin, (center, win-1), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    # Draw Left 
    if feature in ("cross", "tee"):
        endpoint = rotate_point(origin, (1, center), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    # Draw Right 
    if feature in ("cross", "tee", "elbow"):
        endpoint = rotate_point(origin, (win-1, center), angle)
        cv2.line(feature_output, origin, endpoint, (1, 1, 1), thickness)
    
    return feature_output*255


# Feature generation starts here:

# Window Size
win = 64
# Starting Position
deg = 0
# Toggle animation
animation = True
preview = True
loop = False

img_stack = []

if animation:
    while deg in range(360):
        feature = generate_feature(win, "cross", deg, 1)
        img_stack.append(feature)
        if preview:
            cv2.imshow("Feature", feature)
            cv2.waitKey(16)
        if loop:
            if deg == 360:
                deg =  0
            elif deg == 360:
                break
        deg += 1
else:
    feature = generate_feature(win, "cross", deg, 2)
    cv2.imshow("Feature", feature)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

frames = [Image.fromarray(img) for img in img_stack]
frames[0].save('stacked_images.tiff', save_all=True, append_images=frames[1:])
