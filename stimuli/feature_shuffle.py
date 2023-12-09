import csv
import numpy as np
import tifffile

with open("dataset/scatter/labels_scatter.csv", newline='') as csvfile:
    img_names = [row[0] for row in csv.reader(csvfile) if row]

print(img_names[0])
for img in img_names:
    with tifffile.TiffFile(f"dataset/{img}") as tiff:
        tiff_stack = tiff.asarray().astype(np.uint8)

        # Set shuffle offset
        lo = tiff_stack.shape[0]//4
        hi = tiff_stack.shape[0] - tiff_stack.shape[0]//4 
        offset = np.random.randint(lo, hi)

        # Shuffle
        offset_frames = np.concatenate([tiff_stack[offset:,:,:], tiff_stack[:offset,:,:]], axis=0)

        # Save
        tifffile.imwrite(f"dataset/{img}", offset_frames, mode='w')


