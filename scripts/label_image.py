import numpy as np
import cv2 as cv

import os

WIP = False
#img_path = "ignore/WIP_datasets/B/b11.jpg"
#img_path = "ignore/board_clean.png"
board = cv.imread(img_path, 1)

def get_click_data(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN or event == cv.EVENT_MBUTTONDOWN:
        cancel = (event == cv.EVENT_RBUTTONDOWN)
        city = (event == cv.EVENT_MBUTTONDOWN)

        save_info = (cancel, x, y, city)
        param[0] = True # New data available
        param[1] = save_info

cv.namedWindow('display', cv.WINDOW_GUI_NORMAL)

param = [False, None]
cv.setMouseCallback('display', get_click_data, param)

img_root, img_name = os.path.split(img_path)
img_name_noext, _ = os.path.splitext(img_name)

results_file_name = img_name_noext
if WIP:
    results_file_name += "_labels" + "_wip"
results_file_name += ".txt"
results_file_path = os.path.join(img_root, results_file_name)
# Draw already read rectangles on image
if os.path.exists(results_file_path):
    with open(results_file_path, 'r') as results_file:
        lines = [line.rstrip() for line in results_file]
        for line in lines:
            if line[-1] == ',':
                line = line[:-1]

            line_split = line.split(", ")
            if len(line_split) <= 10:
                rectangle_coords = []
                for _ in range(4):
                    x = int(line_split.pop(0))
                    y = int(line_split.pop(0))
                    rectangle_coords.append((x, y))
                for i in range(4):
                    coords = rectangle_coords[i]
                    next_coords = rectangle_coords[(i+1) % 4]
                    cv.line(board, coords, next_coords, (255, 127, 255), 1)
            else:
                octagon_coords = []
                for _ in range(8):
                    x = int(line_split.pop(0))
                    y = int(line_split.pop(0))
                    octagon_coords.append((x, y))
                for i in range(8):
                    coords = octagon_coords[i]
                    next_coords = octagon_coords[(i+1) % 8]
                    cv.line(board, coords, next_coords, (255, 255, 127), 1)

with open(results_file_path, 'a') as results_file:
    while True:
        # Loop for a single rectangle
        rectangle_coords = []
        while len(rectangle_coords) < 4:
            # Get click properties
            param[0] = False
        
            while not param[0]:
                cv.imshow('display', board)
                cv.waitKey(20)

            cancel, x, y, city = param[1]

            if not cancel and not city:
                rectangle_coords.append((x, y))

                board_backup = board.copy()
                if len(rectangle_coords) == 1:
                    radius = 1
                    cv.circle(board, (x, y), radius, (255, 127, 255), -1)
                else:
                    old_x, old_y = rectangle_coords[-2]
                    cv.line(board, (old_x, old_y), (x, y), (255, 127, 255), 1)
                
                if len(rectangle_coords) == 4:
                    cv.line(board, (x, y), rectangle_coords[0], (255, 127, 255), 1)
                    # cv.imshow('display', board)
                    # cv.waitKey(20)
                    # user_input = input("Enter class name: ")
                    # if user_input == "" or user_input == "cancel" or user_input == "back":
                    #     cancel = True
            
            if cancel:
                if len(rectangle_coords) != 0:
                    rectangle_coords.pop()
                board = board_backup

            if city:
                radius = 11
                diag = round(np.sqrt(2)*(radius/2))
                octagon_coords = []
                octagon_coords.append((x, y+radius))
                octagon_coords.append((x+diag, y+diag))
                octagon_coords.append((x+radius, y))
                octagon_coords.append((x+diag, y-diag))
                octagon_coords.append((x, y-radius))
                octagon_coords.append((x-diag, y-diag))
                octagon_coords.append((x-radius, y))
                octagon_coords.append((x-diag, y+diag))

                for i in range(8):
                    coords = octagon_coords[i]
                    next_coords = octagon_coords[(i+1) % 8]
                    cv.line(board, coords, next_coords, (255, 255, 127), 1)

                save_str = ""
                for octagon_coord in octagon_coords:
                    save_str += str(octagon_coord[0]) + ", " + str(octagon_coord[1]) + ", "
                save_str += "city, 0\n"
                results_file.write(save_str)

        save_str = ""
        for rectangle_coord in rectangle_coords:
            save_str += str(rectangle_coord[0]) + ", " + str(rectangle_coord[1]) + ", "

        #save_str += user_input + ", 0"
        save_str += '\n'
        results_file.write(save_str)
