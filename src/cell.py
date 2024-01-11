import numpy as np
import cv2 as cv
import bisect
import math

class Cell:
    def __init__(self, index, rect, rbox, collision_extend_by=35):
        # Rect is (center (x,y), (width, height), angle of rotation radians)
        self.index = index
        self.rect = (np.intp(np.array(rect[0])), np.array(rect[1]), (rect[2]*np.pi)/180)
        self.rbox = rbox
        # Link_a and Link_b. Inside each, links are sorted by closest collision point
        self.links = ([], [])

        self.random_color = Cell.random_color()

        # --------- Compute collision lines ---------

        # A. Compute line inside box

        min_ln_i = np.argmin(self.lengths) # min length index
        max_ln_i = (min_ln_i + 1) % 2 # max length index
        mid_point_a = ((rbox[1+min_ln_i][0] + rbox[2+min_ln_i][0]) // 2, (rbox[1+min_ln_i][1] + rbox[2+min_ln_i][1]) // 2)
        mid_point_b = ((rbox[(3+min_ln_i) % 4][0] + rbox[0+min_ln_i][0]) // 2, (rbox[(3+min_ln_i) % 4][1] + rbox[0+min_ln_i][1]) // 2)

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

    def lines(self):
        yield (self.rbox[0], self.rbox[1])
        yield (self.rbox[1], self.rbox[2])
        yield (self.rbox[2], self.rbox[3])
        yield (self.rbox[3], self.rbox[0])
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
    
    # Not used anymore. Used to help remove false positives on the color tresholding
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
            if len(self.links[side]) > 0 and not isinstance(self.links[side][0][0], Cell):
                continue

            for other_cell in other_cells:
                (collides, intersection_pts) = other_cell.collides_with_line(self_lines[side])

                if collides:
                    other_side = other_cell.add_link(self, side, intersection_pts) 
                    self.add_link(other_cell, other_side, intersection_pts, side) 

    def add_link(self, other_object, other_side, collision_points, link_side=None):
        # Compute collision distances and find the smallest
        collision_dists = []
        for collision_point in collision_points:
            collision_dists.append(np.linalg.norm((self.center[0] - collision_point[0], self.center[1] - collision_point[1])))
        min_collision_dist_index = np.argmin(collision_dists)
        min_collision_dist = collision_dists[min_collision_dist_index]

        # If unknown, compute which side of the cell the collision is on
        if link_side is None:
            # dist_a = np.linalg.norm((self.line_a[0][0] - collision_points[min_collision_dist_index][0], self.line_a[0][1] - collision_points[min_collision_dist_index][1]))
            # dist_b = np.linalg.norm((self.line_b[1][0] - collision_points[min_collision_dist_index][0], self.line_b[1][1] - collision_points[min_collision_dist_index][1]))
            dist_a = np.linalg.norm((self.line_a[0][0] - other_object.center[0], self.line_a[0][1] - other_object.center[1]))
            dist_b = np.linalg.norm((self.line_b[1][0] - other_object.center[0], self.line_b[1][1] - other_object.center[1]))
            link_side = np.argmin((dist_a, dist_b))

        # Don't add new link if the side is already linked with a node
        if len(self.links[link_side]) > 0 and not isinstance(self.links[link_side][0][0], Cell):
            return link_side
        
        # Find other object distance 
        dist = math.dist(self.center, other_object.center)

        # Insert the new link while keeping the list sorted by smallest distance first
        bisect.insort(self.links[link_side], (other_object, other_side, dist), key=lambda x: x[1])

        return link_side
    
    def keep_best_link(self, link_side, best_link_index):
        for i in reversed(range(len(self.links[link_side]))):
            if self.links[link_side][i][0].index != best_link_index:
                del self.links[link_side][i]

    def get_links_dicts(self):
        links_dicts = [{}, {}]
        for links_side, links_dict in zip(self.links, links_dicts):
            for link in links_side:
                obj, obj_side, dist = link

                # If linked object is a node, skip it
                if not isinstance(obj, Cell):
                    continue
                
                links_dict[2*obj.index + obj_side] = dist

        return links_dicts

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
        for (i, (line_pt_1, line_pt_2)) in enumerate(self.lines()):
            if i >= 4:
                break
            cv.line(img, line_pt_1, line_pt_2, (0, 0, 255), 2)
        cv.putText(img, str(self.index), self.center, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 127, 255), 2)

    def draw_hitbox(self, img):
        colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for (i, (line_pt_1, line_pt_2)) in enumerate(self.lines()):
            cv.line(img, line_pt_1, line_pt_2, colors[i], 1)
    
    def draw_links(self, img):
        for links_side in self.links:
            for link in links_side:
                color = self.random_color if isinstance(link[0], Cell) else (255, 63, 63)
                
                halfway_point = np.intp(np.round(np.divide(self.center + link[0].center, 2)))
                cv.line(img, self.center, halfway_point, color, 3)
                cv.line(img, halfway_point, link[0].center, color, 1)

    def random_color():
        random_channels = np.random.randint(low=0, high=256, size=2)
        random_color = np.append(random_channels, 255)
        np.random.shuffle(random_color)
        return tuple([int(c) for c in random_color])

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
