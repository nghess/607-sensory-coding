import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
import tifffile

# Feature List
features = ["cross", "tee", "elbow", "radius", "diameter"]
spin = ["clockwise", "counterclockwise"]
lineweights = [1,4,16]

for shape in features:
    for lw in lineweights:
        for dir in spin:
            # Load 1st-gen feature
            in_fn = f"{shape}_{lw}_{dir}.tiff"
            with tifffile.TiffFile(f"dataset/{in_fn}") as tiff:
                tiff_stack = tiff.asarray().astype(np.uint8)*255

                # Generate blank canvas
                canvas = np.zeros((tiff_stack.shape[0], tiff_stack.shape[1], tiff_stack.shape[2]), np.uint8)

                # Set scale
                divisor = 2
                rescale = 1.0/divisor

                # Rescale imported tiff stack along x and y dimensions
                feature_scaled = scipy.ndimage.zoom(tiff_stack, (1, rescale, rescale), order=3)

                """
                Random window with a center no closer than feature_scaled.shape[0]//2
                """
                ceil = tiff_stack.shape[1]//divisor
                x = int(np.random.uniform(0,ceil))
                y = int(np.random.uniform(0,ceil))
                canvas[:,x:x+ceil,y:y+ceil] += feature_scaled

                # Save the array as a TIFF stack  <----Need to convert to binary
                out_fn = f"{shape}_{lw}_{dir}_x{x}_y{y}_d{rescale}.tiff"
                tifffile.imwrite(f"dataset/scatter/{out_fn}", canvas, mode='w')

"""
Preview Output
"""

# cv2.imshow("Feature", canvas[30, :, :])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Animated Preview
# deg = 0
# while deg in range(tiff_stack.shape[0]):
#     cv2.imshow("Feature", canvas[deg, :, :])
#     cv2.waitKey(1)
#     deg += 1
#     if deg == tiff_stack.shape[0]:
#         deg =  0
    