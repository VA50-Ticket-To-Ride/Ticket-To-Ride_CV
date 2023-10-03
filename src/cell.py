import numpy as np
import cv2 as cv
import bisect

class Cell:
    def __init__(self, id_box, box, color_name, collision_extend_by=72):
        self.id = id_box # To eventually replace by references to other objects
        self.box = box
        self.color_name = color_name
        # Link_a and Link_b. Inside each, links are sorted by closest collision point
        self.links = ([], [])

        # --------- Compute center ---------

        self.center = ((box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2)

        # --------- Compute collision lines ---------

        # A. Compute line inside box

        box_length1 = np.linalg.norm(box[0] - box[1])
        box_length2 = np.linalg.norm(box[1] - box[2])
        box_lengths = ((box_length1, box_length2))
        min_ln_i = np.argmin(box_lengths) # min length index
        max_ln_i = (min_ln_i + 1) % 2 # max length index
        mid_point_a = ((box[0+min_ln_i][0] + box[1+min_ln_i][0]) // 2, (box[0+min_ln_i][1] + box[1+min_ln_i][1]) // 2)
        mid_point_b = ((box[2+min_ln_i][0] + box[(3+min_ln_i) % 4][0]) // 2, (box[2+min_ln_i][1] + box[(3+min_ln_i) % 4][1]) // 2)

        # B. Extend line and split it

        a_extended = [0, 0]
        a_extended[0] = mid_point_a[0] + ((mid_point_a[0] - mid_point_b[0]) / (box_lengths[max_ln_i])) * collision_extend_by
        a_extended[1] = mid_point_a[1] + ((mid_point_a[1] - mid_point_b[1]) / (box_lengths[max_ln_i])) * collision_extend_by
        a_extended = np.intp(a_extended)

        b_extended = [0, 0]
        b_extended[0] = mid_point_b[0] + ((mid_point_b[0] - mid_point_a[0]) / (box_lengths[max_ln_i])) * collision_extend_by
        b_extended[1] = mid_point_b[1] + ((mid_point_b[1] - mid_point_a[1]) / (box_lengths[max_ln_i])) * collision_extend_by
        b_extended = np.intp(b_extended)
        
        self.line_a = (a_extended, self.center)
        self.line_b = (self.center, b_extended)

        # --------- Is this a tunnel? ---------

        self.is_tunnel = (box_lengths[max_ln_i] < 100)

    def lines(self):
        yield (self.box[0], self.box[1])
        yield (self.box[1], self.box[2])
        yield (self.box[2], self.box[3])
        yield (self.box[3], self.box[0])
        yield self.line_a
        yield self.line_b

    def collides_with(self, line):
        collides = False
        intersection_pts = []
        for internal_line in self.lines():
            line_collides, intersection_pt = Cell.test_collision(internal_line, line)
            if line_collides:
                collides = True
                intersection_pts.append(intersection_pt)
            
        return (collides, intersection_pts)
    
    def search_collisions(self, other_cells):
        # Two directions to test collisions for
        self_lines = (self.line_a, self.line_b)
        for i in range(2):
            for other_cell in other_cells:
                (collides, intersection_pts) = other_cell.collides_with(self_lines[i])

                if collides:
                    other_cell.add_link(self.id, intersection_pts) 
                    self.add_link(other_cell.id, intersection_pts, i) 

    def add_link(self, other_id, collision_points, link_index=-1):
        collision_dists = []
        for collision_point in collision_points:
            collision_dists.append(np.linalg.norm((self.center[0] - collision_point[0], self.center[1] - collision_point[1])))
        min_collision_dist_index = np.argmin(collision_dists)
        min_collision_dist = collision_dists[min_collision_dist_index]

        # If unknown, compute which side of the cell the collision is on
        if link_index < 0:
            dist_a = np.linalg.norm((self.line_a[0][0] - collision_points[min_collision_dist_index][0], self.line_a[0][1] - collision_points[min_collision_dist_index][1]))
            dist_b = np.linalg.norm((self.line_b[1][0] - collision_points[min_collision_dist_index][0], self.line_b[1][1] - collision_points[min_collision_dist_index][1]))
            link_index = np.argmin((dist_a, dist_b))

        # Insert the new link while keeping the list sorted by closest collision first
        bisect.insort(self.links[link_index], (other_id, min_collision_dist), key=lambda x: x[1])

    def draw(self, img):
        colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for (i, (line_pt_1, line_pt_2)) in enumerate(self.lines()):
            cv.line(img, line_pt_1, line_pt_2, colors[i], 2)

        cv.putText(img, str(self.id), self.center, cv.FONT_HERSHEY_SIMPLEX, 1.6, (255, 128, 255), 4)

    def test_collision(line1, line2):
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
