import math
import numpy as np
import cv2 as cv

class TrainTrack:
    def __init__(self, cell):
        self.cells = [cell]
        self.links = []

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_link(self, link):
        self.links.append(link)

    def draw(self, img):
        color = self.cells[0].random_color

        # Find out which cell is the closest to the first link
        dists = [math.dist(cell.center, self.links[0].node.center) for cell in self.cells]
        cells_iterator = iter(self.cells) if np.argmin(dists) == 0 else reversed(self.cells)
        
        # Draw lines
        old_cell = self.links[0].node
        for cell in cells_iterator:
            cv.line(img, cell.center, old_cell.center, color, 3)
            old_cell = cell

        # Draw last line
        if len(self.links) > 1:
            cv.line(img, old_cell.center, self.links[1].node.center, color, 3)

