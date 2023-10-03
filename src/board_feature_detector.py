from cell import *

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

    blue_lower_gimp = (195, 50, 65)
    blue_upper_gimp = (215, 255, 255)

    # WIP
    gray_lower_gimp = (0, 0, 40)
    gray_upper_gimp = (179, 5.5, 65)

    def __init__(self, board):
        # Convert BGR to HSV
        # https://stackoverflow.com/questions/53977777/how-can-i-only-keep-text-with-specific-color-from-image-via-opencv-and-python
        board_hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

        # BLUE
        # Define treshold HSV values
        hsv_lower = BoardFeatureDetector.hsv_gimp_to_cv(BoardFeatureDetector.blue_lower_gimp)
        hsv_upper = BoardFeatureDetector.hsv_gimp_to_cv(BoardFeatureDetector.blue_upper_gimp)
        board_mask = BoardFeatureDetector.treshold_color_simple(board_hsv, hsv_lower, hsv_upper)

        # Detect cells
        blue_cells = BoardFeatureDetector.detect_cells(board_mask, "blue")

        # Build collision map
        for (i, cell) in enumerate(blue_cells):
            cell.search_collisions(blue_cells[i+1:])

        # Debug print
        if True:
            board_mask_bgr = cv.cvtColor(board_mask, cv.COLOR_GRAY2BGR)
            for cell in blue_cells:
                cell.draw(board_mask_bgr)
                print("ID: " + str(cell.id))
                print("Green: " + str(cell.links[0]))
                print("Blue: " + str(cell.links[1]))
                print()
            
            out = board_mask_bgr

            cv.namedWindow('out', cv.WINDOW_NORMAL)
            cv.imshow('out', out)

            cv.resizeWindow('out', 1920, 1080)

            cv.waitKey(0)

    def treshold_color_simple(img_hsv, hsv_lower, hsv_upper):
        # Threshold the HSV image to get only target color
        img_mask = cv.inRange(img_hsv, hsv_lower, hsv_upper)

        # Think upon doing it the same as the blob detection algorithm, with multiple tresholds and deltas (may help with gray)
        morph_kernel = np.ones((3, 3), np.uint8)
        #board_mask_blue_morphed = cv.erode(board_mask_blue, morph_kernel, iterations = 1)
        img_mask_morphed = cv.morphologyEx(img_mask, cv.MORPH_OPEN, morph_kernel)
        img_mask_morphed = cv.dilate(img_mask_morphed, morph_kernel, iterations = 2)

        # Bitwise-AND mask and original image
        #board_blue = cv.bitwise_and(board, board, mask=board_mask)

        return img_mask_morphed

    def detect_cells(img_mask, color_name):
        # Keep external contours only
        contours, _ = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Detect cells
        cell_id = 0
        cells = []
        for cnt in contours:
            # Find the smallest rectangle fitting the contour
            rect = cv.minAreaRect(cnt)

            # Turn the rotated rectangle into coordinates
            box = cv.boxPoints(rect)
            
            # Round values
            box = np.intp(box)

            cell = Cell(cell_id, box, color_name)
            cells.append(cell)

            cell_id += 1

        return cells

    def hsv_gimp_to_cv(hsv_gimp):
        # OpenCV HSV range is [179, 255, 255] where gimp is [359, 100, 100]
        hsv_cv = np.array([hsv_gimp[0]/2, (hsv_gimp[1]*255)/100, (hsv_gimp[2]*255)/100])
        return hsv_cv
