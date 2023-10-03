from board_feature_detector import *

img_name = "ignore/board_2.png"
board = cv.imread(img_name, 1)
assert board is not None, "Board image could not be read, check with os.path.exists()"

BoardFeatureDetector(board)

"""
Unused idea, image contrast boost

lab = cv.cvtColor(board, cv.COLOR_BGR2LAB)
l_channel, a, b = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l_channel)

limg = cv.merge((cl, a, b))
enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#enhanced_img = board_smooth

board_smooth = cv.GaussianBlur(enhanced_img, (9, 9), 0)


board_smooth_grayscale = cv.cvtColor(board_smooth, cv.COLOR_BGR2GRAY)
edges = cv.Canny(board_smooth_grayscale, 100, 180)
out = edges
"""
