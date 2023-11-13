from cell import *
from node import *

import math

class BoardFeatureDetector:
    """
    Docs dump:
    Contours detection
    https://docs.opencv.org/4.8.0/d4/d73/tutorial_py_contours_begin.html
    https://docs.opencv.org/4.8.0/dd/d49/tutorial_py_contour_features.html
    https://docs.opencv.org/4.8.0/d9/d8b/tutorial_py_contours_hierarchy.html

    Collision
    https://stackoverflow.com/questions/56100547/how-do-i-check-collision-between-a-line-and-a-rect-in-pygame
    """
    def __init__(self, board):
        # Convert BGR to HSV
        # https://stackoverflow.com/questions/53977777/how-can-i-only-keep-text-with-specific-color-from-image-via-opencv-and-python
        board_hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

        # Define treshold HSV values
        color_treshs = {}

        blue_lower_gimp = (195, 50, 65)
        blue_upper_gimp = (215, 100, 100)
        color_treshs["blue"] = (BoardFeatureDetector.hsv_gimp_to_cv(blue_lower_gimp), 
            BoardFeatureDetector.hsv_gimp_to_cv(blue_upper_gimp))

        # WIP
        orange_lower_gimp = (25, 50, 65)
        orange_upper_gimp = (40, 100, 100)
        color_treshs["orange"] = (BoardFeatureDetector.hsv_gimp_to_cv(orange_lower_gimp), 
            BoardFeatureDetector.hsv_gimp_to_cv(orange_upper_gimp))

        # WIP
        gray_lower_gimp = (0, 0, 40)
        gray_upper_gimp = (179, 5.5, 65)
        color_treshs["gray"] = (BoardFeatureDetector.hsv_gimp_to_cv(gray_lower_gimp), 
            BoardFeatureDetector.hsv_gimp_to_cv(gray_upper_gimp))
        
        # ELEMENT DETECTION
        cells = {}

        # BLUE
        board_mask = BoardFeatureDetector.treshold_color_simple(board_hsv, color_treshs["blue"][0], color_treshs["blue"][1])
        # Detect cells
        cells["blue"] = BoardFeatureDetector.detect_cells(board_mask, "blue")
        
        # ORANGE
        board_mask = BoardFeatureDetector.treshold_color_simple(board_hsv, color_treshs["orange"][0], color_treshs["orange"][1])
        # Detect nodes and cells
        nodes = BoardFeatureDetector.detect_nodes(board_mask)
        cells["orange"] = BoardFeatureDetector.detect_cells(board_mask, "orange")

        # Remove cells mostly inside nodes
        for node in nodes:
            for cells_color in cells.values():
                # See which cells need removing
                indexes_to_remove = []
                for i, cell in enumerate(cells_color):
                    if node.is_cell_mostly_inside(cell):
                        indexes_to_remove.append(i)

                # Remove duplicate indexes
                indexes_to_remove = set(indexes_to_remove)

                # Actually remove cells
                for index in sorted(indexes_to_remove, reverse=True):
                    del cells_color[index]

        #"""
        # Build collision map
        for node in nodes:
            node.search_collisions(cells)
            
        for cells_color in cells.values():
            for (i, cell) in enumerate(cells_color):
                cell.search_collisions(cells_color[i+1:])
        #"""

        # Debug print
        if True:
            board_mask_bgr = cv.cvtColor(board_mask, cv.COLOR_GRAY2BGR)
            board_mask_bgr = board
            
            #"""
            #board_mask_bgr = cv.drawContours(board_mask_bgr, contours, -1, (0,255,0), 3)
            for node in nodes:
                node.draw(board_mask_bgr)
                #circle_area = (math.pi*radius)*(math.pi*radius)
                #cv.putText(board_mask_bgr, str(radius), center, cv.FONT_HERSHEY_SIMPLEX, 1.6, (255, 128, 255), 4)
            #"""

            #"""
            for cells_color in cells.values():
                for cell in cells_color:
                    cell.draw(board_mask_bgr)
                    #cv.drawContours(board_mask_bgr, [cell.box], -1, (0,255,0), 3)

                    # print("ID: " + str(cell.id))
                    # print("Green: " + str(cell.links[0]))
                    # print("Blue: " + str(cell.links[1]))
                    # print()
            #"""
            
            out = board_mask_bgr

            cv.namedWindow('out', cv.WINDOW_NORMAL)
            cv.imshow('out', out)

            cv.resizeWindow('out', 1920, 1080)

            cv.waitKey(0)

    def treshold_color_simple(img_hsv, hsv_lower, hsv_upper):
        # Threshold the HSV image to get only target color
        img_mask = cv.inRange(img_hsv, hsv_lower, hsv_upper)

        """
        # Think upon doing it the same as the blob detection algorithm, with multiple tresholds and deltas (may help with gray)
        morph_kernel = np.ones((3, 3), np.uint8)
        #board_mask_blue_morphed = cv.erode(board_mask_blue, morph_kernel, iterations = 1)
        img_mask_morphed = cv.morphologyEx(img_mask, cv.MORPH_OPEN, morph_kernel)
        img_mask_morphed = cv.dilate(img_mask_morphed, morph_kernel, iterations = 2)
        """

        # Think upon doing it the same as the blob detection algorithm, with multiple tresholds and deltas (may help with gray)
        #morph_kernel = np.array(([0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]), np.uint8)
        morph_kernel = np.ones((5, 5), np.uint8)
        #img_mask_morphed = cv.erode(img_mask, morph_kernel, iterations = 1)
        img_mask_morphed = cv.morphologyEx(img_mask, cv.MORPH_OPEN, morph_kernel)
        #img_mask_morphed = cv.dilate(img_mask_morphed, morph_kernel, iterations = 3)

        # Bitwise-AND mask and original image
        #board_blue = cv.bitwise_and(board, board, mask=board_mask)

        return img_mask_morphed
    
    def get_oriented_length(rect, other_rect):
        # Find the length of the rectangle most oriented towards the center of the other rectangle
        # Rect are (center (x,y), (width, height), angle of rotation degrees)

        # Calculate the vector from the center of the other rectangle to the center of the current rectangle
        relative_vector = np.array(rect[0]) - np.array(other_rect[0])

        # Calculate the angle between the vector and the x-axis
        angle_to_x_axis = np.arctan2(relative_vector[1], relative_vector[0])

        # Calculate the angle difference between the orientation of the rectangle and the angle to the other rectangle
        angle_difference = np.abs(((rect[2]*np.pi)/180) - angle_to_x_axis)

        # Ensure the angle difference is within the range [0, pi]
        angle_difference = np.minimum(angle_difference, np.pi - angle_difference)

        # Determine which side of the rectangle is most oriented towards the other rectangle
        if angle_difference < np.pi / 4:
            # Width is most oriented
            return rect[1][0]
        else:
            # Height is most oriented
            return rect[1][1]
        
    def detect_cells(img_mask, color_name):
        # Keep external contours only
        contours, _ = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # BoardFeatureDetector.display_contours(img_mask, contours)
        # BoardFeatureDetector.display_contours(img_mask, contours, one_by_one=True)

        # Detect cells
        cells = []
        remaining_rects = []
        for cnt in contours:
            # Find the smallest rectangle fitting the contour
            rect = cv.minAreaRect(cnt)

            # Only keep rectangles whose smallest side is between 25 and 50
            smallest_length = min(rect[1])
            if smallest_length < 25 or smallest_length > 50:
                continue
            
            # Turn the rotated rectangle into coordinates
            box = cv.boxPoints(rect)
            
            # If a rectangle has a longest side smaller than 100, it's either a half or a tunnel. Store it away
            longest_length = max(rect[1])
            if longest_length < 100:
                remaining_rects.append((rect, box))
                continue

            # Round values
            box = np.intp(box)

            # print(rect)
            # BoardFeatureDetector.display_contours(img_mask, [box])

            cell = Cell(box, rect[0], rect[1], color_name)
            cells.append(cell)

        # Parse remaining rectangles (too small to have been validated on first pass)
        # The goal is to find rectangle halves and fuse them together
        while len(remaining_rects) != 0:
            rect = remaining_rects[0]
            box = np.intp(rect[1])

            # Skip all gap computing if there is a single rect left
            if len(remaining_rects) > 1:

                # Find closest rect
                closest_rect = None
                closest_rect_index = None
                closest_dist = None
                for i, other_rect in enumerate(remaining_rects[1:], 1):
                    dist = np.linalg.norm((rect[0][0][0] - other_rect[0][0][0], rect[0][0][1] - other_rect[0][0][1]))

                    if (closest_dist is None) or (dist < closest_dist):
                        closest_rect = other_rect
                        closest_rect_index = i
                        closest_dist = dist

                # Get oriented lengths for both rectangles
                oriented_length = BoardFeatureDetector.get_oriented_length(rect[0], closest_rect[0])
                other_oriented_length = BoardFeatureDetector.get_oriented_length(closest_rect[0], rect[0])

                # Get approximate gap size between the two rectangles by the difference between 
                # the distances and the oriented lengths
                gap = closest_dist - (oriented_length/2) - (other_oriented_length/2)
                
                other_box = np.intp(closest_rect[1])
                
                # Print gap and pair
                # print(gap)
                # BoardFeatureDetector.display_contours(img_mask, [box, other_box])
            
            # The gap is too big, consider this a rectangle by itself (probably a tunnel)
            if len(remaining_rects) == 1 or gap > 20:
                new_rect = rect[0]
                new_box = box
                remaining_rects.pop(0)

            # Else, the gap is small enough, these are a single rectangle
            else:
                # Find the smallest rectangle fitting both
                new_rect = cv.minAreaRect(np.concatenate((box, other_box)))
                new_box = cv.boxPoints(new_rect)
                new_box = np.intp(new_box)
                
                # Remove the first element and the other's index
                remaining_rects.pop(closest_rect_index)
                remaining_rects.pop(0)

            cell = Cell(new_box, new_rect[0], new_rect[1], color_name)
            cells.append(cell)

        return cells
    
    def detect_nodes(img_mask):
        # Keep external contours only
        contours, _ = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Detect circles
        circles = []
        for cnt in contours:
            # Find the smallest circle fitting the contour
            (x,y), radius = cv.minEnclosingCircle(cnt)

            # Discard circles not of fitting size
            if radius < 8 or radius > 22:
                continue
            
            # Discard circles where the ratio of contour area over circle area is too small
            #area = cv.contourArea(cnt)

            circles.append(((x,y), radius))
        
        # Group circles close to each other

        nb_groups = 0
        groups = []
        group_ids = [None]*len(circles)
        for (i, circle) in enumerate(circles):
            (x_a, y_a), radius_a = circle
            
            group_index = nb_groups
            if group_ids[i] is None:
                group_ids[i] = group_index
                groups.append([])
                nb_groups += 1
            else:
                group_index = group_ids[i]

            groups[group_index].append(circle)

            for other_i, other_cell in enumerate(circles[i+1:], start=i+1):
                (x_b, y_b), radius_b = other_cell
                dist = math.sqrt((x_b - x_a)*(x_b - x_a) + (y_b - y_a)*(y_b - y_a))
                if dist < 80:
                    group_ids[other_i] = group_index

        # Display single cell
        # cell_mask = np.zeros_like(img_mask)
        # cell_mask = cv.circle(cell_mask, (x,y), radius, (255, 255, 255), -1)
        
        # Find center of mass of each group and create a nod out of it
        nodes = [None] * len(groups)
        for i, group in enumerate(groups):
            new_x = group[0][0][0]
            new_y = group[0][0][1]
            new_radius = 80
            
            if len(group) > 1:
                sum_x = 0
                sum_y = 0
                total_weights = 0
                for (x,y),radius in group:
                    sum_x += x * radius
                    sum_y += y * radius
                    total_weights += radius
                new_x = sum_x / total_weights
                new_y = sum_y / total_weights
            
            # Create node
            nodes[i] = Node((round(new_x), round(new_y)), new_radius)

        return nodes

    def hsv_gimp_to_cv(hsv_gimp):
        # OpenCV HSV range is [179, 255, 255] where gimp is [359, 100, 100]
        hsv_cv = np.array([hsv_gimp[0]/2, (hsv_gimp[1]*255)/100, (hsv_gimp[2]*255)/100])
        return hsv_cv
    
    def display_contours(img, contours, one_by_one=False):
        img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        print(contours)
        
        if one_by_one:
            for cnt in contours:
                img_bgr_copy = np.copy(img_bgr)

                cv.drawContours(img_bgr_copy, cnt, -1, (0,255,0), 3)

                cv.namedWindow('out', cv.WINDOW_NORMAL)
                cv.imshow('out', img_bgr_copy)
                cv.resizeWindow('out', 1920, 1080)
                cv.waitKey(0)
        
        else:
            cv.drawContours(img_bgr, contours, -1, (0,255,0), 3)
            
            cv.namedWindow('out', cv.WINDOW_NORMAL)
            cv.imshow('out', img_bgr)
            cv.resizeWindow('out', 1920, 1080)
            cv.waitKey(0)
