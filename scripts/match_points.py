import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment

import os

# Read main dataset
main_path = "ignore/WIP_datasets/B/b0.txt"
wip_name_noext = "b11"
main_root, _ = os.path.split(main_path)
wip_path = os.path.join(main_root, wip_name_noext + "_labels_wip.txt")
wip_img_path = os.path.join(main_root, wip_name_noext + ".jpg")
target_path = os.path.join(main_root, wip_name_noext + ".txt")

def get_centers(file_path):
    with open(file_path, 'r') as label_file:
        lines = [line.rstrip() for line in label_file]
        centers = []
        for line in lines:
            if line[-1] == ',':
                line = line[:-1]
            line_split = line.split(", ")
            if len(line_split) > 10:
                continue
            rectangle_coords = []
            for _ in range(4):
                x = int(line_split.pop(0))
                y = int(line_split.pop(0))
                rectangle_coords.append((x, y))

            # Compute center
            center_x = sum(x for x, _ in rectangle_coords) / len(rectangle_coords)
            center_y = sum(y for _, y in rectangle_coords) / len(rectangle_coords)
            centers.append((center_x, center_y))

    assert len(centers) == 300
    return centers

main_centers = np.array(get_centers(main_path))
wip_centers = np.array(get_centers(wip_path))

# Calculate distances between all points in both sets
distances = np.linalg.norm(main_centers[:, None] - wip_centers, axis=2)

# Apply the Hungarian algorithm to minimize the total distance
row_ind, col_ind = linear_sum_assignment(distances)
assert all(ind < 300 for ind in row_ind)
# for i, j in zip(row_ind, col_ind):
#     print(str(i) + " " + str(j))

# Pair the points based on the indices found by the Hungarian algorithm
target_to_main = {}
for i, j in zip(row_ind, col_ind):
    target_to_main[j] = i

wip_img = cv.imread(wip_img_path, 1)

# Read classes
classes = []
with open(main_path, 'r') as label_file:
    lines = [line.rstrip() for line in label_file]
    for line in lines:
        line_split = line.split(", ")
        if len(line_split) <= 10:
            classes.append(line_split[8])

for b_ind, a_ind in target_to_main.items():
    a = main_centers[a_ind]
    b = wip_centers[b_ind]
    x_a = round(a[0])
    y_a = round(a[1])
    x_b = round(b[0])
    y_b = round(b[1])
    cv.line(wip_img, (x_a, y_a), (x_b, y_b), (127, 255, 255), 3)
    cv.putText(wip_img, classes[target_to_main[b_ind]], (x_b+10, y_b+10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 127), 1)

cv.namedWindow('display', cv.WINDOW_GUI_NORMAL)
cv.imshow('display', wip_img)
cv.waitKey(0)

if os.path.exists(target_path):
    raise Exception("Safety exit: attempt to overwrite an existing label file")

# Write updated labels file
with open(target_path, 'w') as target_file:
    with open(wip_path, 'r') as label_file:
        lines = [line.rstrip() for line in label_file]
        for i, line in enumerate(lines):
            target_class = classes[target_to_main[i]]
            save_str = line + " " + target_class + ", " + "0\n"
            target_file.write(save_str)
