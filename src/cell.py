import numpy as np
import cv2 as cv
import bisect
import copy

class Cell:
    def __init__(self, rect, color_name, collision_extend_by=72):
        # Turn the rotated rectangle into coordinates
        box = cv.boxPoints(rect)
        # Round values
        self.box = np.intp(box)
        
        # Rect is (center (x,y), (width, height), angle of rotation radians)
        self.rect = (np.intp(np.array(rect[0])), np.array(rect[1]), (rect[2]*np.pi)/180)
        self.color_name = color_name
        # Link_a and Link_b. Inside each, links are sorted by closest collision point
        self.links = ([], [])

        # --------- Compute collision lines ---------

        # A. Compute line inside box

        min_ln_i = np.argmin(self.lengths) # min length index
        max_ln_i = (min_ln_i + 1) % 2 # max length index
        mid_point_a = ((box[1+min_ln_i][0] + box[2+min_ln_i][0]) // 2, (box[1+min_ln_i][1] + box[2+min_ln_i][1]) // 2)
        mid_point_b = ((box[(3+min_ln_i) % 4][0] + box[0+min_ln_i][0]) // 2, (box[(3+min_ln_i) % 4][1] + box[0+min_ln_i][1]) // 2)

        # B. Extend line and split it

        a_extended = [0, 0]
        a_extended[0] = mid_point_a[0] + ((mid_point_a[0] - mid_point_b[0]) / (self.lengths[max_ln_i])) * collision_extend_by
        a_extended[1] = mid_point_a[1] + ((mid_point_a[1] - mid_point_b[1]) / (self.lengths[max_ln_i])) * collision_extend_by
        a_extended = np.intp(a_extended)

        b_extended = [0, 0]
        b_extended[0] = mid_point_b[0] + ((mid_point_b[0] - mid_point_a[0]) / (self.lengths[max_ln_i])) * collision_extend_by
        b_extended[1] = mid_point_b[1] + ((mid_point_b[1] - mid_point_a[1]) / (self.lengths[max_ln_i])) * collision_extend_by
        b_extended = np.intp(b_extended)
        
        self.line_a = (a_extended, self.center)
        self.line_b = (self.center, b_extended)

        # --------- Is this a tunnel? ---------

        self.is_tunnel = (self.lengths[max_ln_i] < 100)

    def lines(self):
        yield (self.box[0], self.box[1])
        yield (self.box[1], self.box[2])
        yield (self.box[2], self.box[3])
        yield (self.box[3], self.box[0])
        yield self.line_a
        yield self.line_b

    def collides_with_point(self, point):
        # Translate the point and rectangle to the origin
        translated_point = point - self.center
        rotated_point = np.dot(translated_point, np.array([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]]))

        half_size = self.lengths / 2

        # Check if the rotated point is inside the rectangle
        return np.abs(rotated_point[0]) <= half_size[0] and np.abs(rotated_point[1]) <= half_size[1]

    def collides_with_line(self, line):
        """
        Checks for line collisions (if the line is fully inside it doesn't count as a collision)
        """
        collides = False
        intersection_pts = []
        for internal_line in self.lines():
            line_collides, intersection_pt = Cell.line_to_line_collision(internal_line, line)
            if line_collides:
                collides = True
                intersection_pts.append(intersection_pt)
            
        return (collides, intersection_pts)
    
    def is_box_mostly_inside(self, box):
        # Create a grid of nine points from box
        grid = list(box)
        for i, point in enumerate(box):
            next_point = box[(i+1) % 4]
            grid.append(((point[0] + next_point[0]) / 2, (point[1] + next_point[1]) / 2))
        grid.append(((box[0][0] + box[2][0]) / 2, (box[0][1] + box[2][1]) / 2))
        
        inside_count = 0
        for point in grid:
            if self.collides_with_point(point):
                inside_count += 1
        
        # A box is counted mostly inside if it has at least 3 points inside
        return inside_count >= 3
    
    def search_collisions(self, other_cells):
        # Two directions to test collisions for
        self_lines = (self.line_a, self.line_b)
        for side in range(2):
            # If there is already a collision with a node, it takes priority ; skip checks
            if len(self.links[side]) > 0:
                continue

            for other_cell in other_cells:
                (collides, intersection_pts) = other_cell.collides_with_line(self_lines[side])

                if collides:
                    other_cell.add_link(self, intersection_pts) 
                    self.add_link(other_cell, intersection_pts, side) 

    def add_link(self, other_object, collision_points, link_side=None):
        # Compute collision distances and find the smallest
        collision_dists = []
        for collision_point in collision_points:
            collision_dists.append(np.linalg.norm((self.center[0] - collision_point[0], self.center[1] - collision_point[1])))
        min_collision_dist_index = np.argmin(collision_dists)
        min_collision_dist = collision_dists[min_collision_dist_index]

        # If unknown, compute which side of the cell the collision is on
        if link_side is None:
            dist_a = np.linalg.norm((self.line_a[0][0] - collision_points[min_collision_dist_index][0], self.line_a[0][1] - collision_points[min_collision_dist_index][1]))
            dist_b = np.linalg.norm((self.line_b[1][0] - collision_points[min_collision_dist_index][0], self.line_b[1][1] - collision_points[min_collision_dist_index][1]))
            link_side = np.argmin((dist_a, dist_b))

        # Insert the new link while keeping the list sorted by closest collision first
        bisect.insort(self.links[link_side], (other_object, min_collision_dist), key=lambda x: x[1])

    @property
    def center(self):
        return self.rect[0]
    
    @property
    def lengths(self):
        return self.rect[1]
    
    @property
    def angle(self):
        # The angle is in radians
        return self.rect[2]
    
    def draw(self, img):
        # colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        # for (i, (line_pt_1, line_pt_2)) in enumerate(self.lines()):
        #     cv.line(img, line_pt_1, line_pt_2, colors[i], 2)

        # cv.putText(img, self.color_name, self.center, cv.FONT_HERSHEY_SIMPLEX, 1.6, (255, 127, 255), 4)

        for links_side in self.links:
            for link in links_side:
                cv.line(img, self.center, link[0].center, (255, 127, 127), 3)

    def line_to_line_collision(line1, line2):
        x1 = line1[0][0]
        y1 = line1[0][1]
        x2 = line1[1][0]
        y2 = line1[1][1]
        x3 = line2[0][0]
        y3 = line2[0][1]
        x4 = line2[1][0]
        y4 = line2[1][1]

        det = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if det == 0:
            # Parallel lines: no intersection
            return (False, (-1, -1))
        
        # Calculate distance to the intersection point
        uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / det
        uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / det

        if (uA >= 0) and (uA <= 1) and (uB >= 0 and uB <= 1):
            # Lines are colliding
            intersection_x = x1 + (uA * (x2-x1))
            intersection_y = y1 + (uA * (y2-y1))
            return (True, (intersection_x, intersection_y))
        
        return (False, (-1, -1))
