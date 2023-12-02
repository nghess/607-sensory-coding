import cv2
import numpy as np
from PIL import Image

def rotate_point(origin, point, angle, direction):

    assert direction in ("clockwise", "counterclockwise"), "Direction argument must either be 'clockwise' or 'counterclockwise'"

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    if direction == "clockwise":
        return int(qx), int(qy)
    elif direction == "counterclockwise":
        return int(qx), int(qy)


# Generates feature to animate
def generate_feature(win, feature, angle=0, direction="clockwise", thickness=1):

    # Check that feature is correctly specified
    assert feature in ("cross", "tee", "elbow", "radius", "diameter"), "Feature must be one of 'cross', 'tee', 'radius', 'diameter'."

    # Convert angle to radians and set direction
    if direction == "clockwise":
        angle = np.deg2rad(angle)
    elif direction == "counterclockwise":
        angle = -np.deg2rad(angle)

    # Define center of feature_output
    center = win // 2
    origin = (center, center)

    # Draw Top
    if feature in ("cross", "tee", "elbow", "radius", "diameter"):
        endpoint = rotate_point(origin, (center, 10), angle, direction)
        cv2.line(feature_output, origin, endpoint, (255,255,255), thickness)
    # Draw Bottom
    if feature in ("cross", "diameter"):
        endpoint = rotate_point(origin, (center, win-10), angle, direction)
        cv2.line(feature_output, origin, endpoint, (255,255,255), thickness)
    # Draw Left 
    if feature in ("cross", "tee"):
        endpoint = rotate_point(origin, (10, center), angle, direction)
        cv2.line(feature_output, origin, endpoint, (255,255,255), thickness)
    # Draw Right 
    if feature in ("cross", "tee", "elbow"):
        endpoint = rotate_point(origin, (win-10, center), angle, direction)
        cv2.line(feature_output, origin, endpoint, (255,255,255), thickness)
    
    return feature_output


'''
Feature generation starts here:
'''

# Window Size
win = 256

# Feature List
features = ["cross", "tee", "elbow", "radius", "diameter"]
spin = ["clockwise", "counterclockwise"]
lineweights = [1,2,4,8]

# Toggle animation and preview
animation = True
preview = False
loop = False
save = True

# Production loop
for shape in features:
    for lw in lineweights:
        for dir in spin:

            # Clear angle and image stack
            deg =  0  
            img_stack = []
            
            # Spin feature and render
            while deg < 360:
                feature = generate_feature(win, shape, deg, direction=dir, thickness=lw)
                img_stack.append(feature)
                
                # For preview visualization
                if preview:
                    cv2.imshow("Feature", feature)
                    cv2.waitKey(1)
                if loop:
                    if deg == 360:
                        deg =  0

                # Increment angle    
                deg += 1

            # Write to Tiff Stack
            if save and not loop:
                frames = [Image.fromarray(img).convert('1') for img in img_stack]
                frames[0].save(f"output/{shape}_{lw}_{dir}.tiff", save_all=True, append_images=frames[1:])
