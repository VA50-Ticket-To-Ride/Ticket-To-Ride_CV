from board_feature_detector import *

img_name = "ignore/datasets/dataset_boards/images/b13.jpg"
board = cv.imread(img_name, 1)
assert board is not None, "Board image could not be read, check with os.path.exists()"

BoardFeatureDetector(board, debug=True)
