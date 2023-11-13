import numpy as np
import cv2 as cv
import math

class Node:
    def __init__(self, center, collision_radius=80):
        self.center = center
        self.collision_radius = collision_radius
        self.links = []

    def is_cell_mostly_inside(self, cell):
        inside_count = 0
        for point in cell.box:
            if self.collides_with_point(point):
                inside_count += 1
        
        # A cell is counted mostly inside if it has at least 3 points inside
        return inside_count >= 3
    
    def search_collisions(self, cells):
        for cells_color in cells.values():
            for cell in cells_color:
                for line in cell.lines():
                    (collides, intersection_pt) = self.collides_with_line(line)

                    if collides:
                        cell.add_link(self, [intersection_pt])
                        self.links.append(cell)
                        break

    def collides_with_line(self, line):
        # See https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
        
        line_start = np.array([line[0][0], line[0][1]], dtype=np.float32)
        line_end = np.array([line[1][0], line[1][1]], dtype=np.float32)
        circle_center = np.array(self.center, dtype=np.float32)

        d = line_end - line_start # Direction vector of line from start to end
        f = line_start - circle_center

        # Solving  (t^2)*(d·d) + 2t*(f·d) + (f·f - r^2) = 0

        a = np.dot(d, d)
        b = 2*np.dot(f, d)
        c = np.dot(f, f) - self.collision_radius**2

        discriminant = b*b-4*a*c

        if (discriminant < 0):
            # No intersection
            return (False, (-1, -1))
        
        # Ray didn't totally miss sphere, so there is a solution to the equation

        discriminant = math.sqrt(discriminant)

        # Either solution may be on or off the ray so need to test both t1 is always 
        # the smaller value, because BOTH discriminant and a are nonnegative.

        t1 = (-b - discriminant)/(2*a)
        t2 = (-b + discriminant)/(2*a)

        # 3x HIT cases:
        #          -o->             --|-->  |            |  --|->
        # Impale(t1 hit,t2 hit), Poke(t1 hit,t2>1), ExitWound(t1<0, t2 hit), 

        # 3x MISS cases:
        #       ->  o                     o ->              | -> |
        # FallShort (t1>1,t2>1), Past (t1<0,t2<0), CompletelyInside(t1<0, t2>1)

        if t1 >= 0 and t1 <= 1:
            # t1 is the intersection, and it's closer than t2 (since t1 uses -b - discriminant)
            # Impale, Poke
            intersection_x = line_start[0] + t1*d[0]
            intersection_y = line_start[1] + t1*d[1]
            return (True, (intersection_x, intersection_y))
        
        # Here t1 didn't intersect so we are either started
        # inside the sphere or completely past it
        if t2 >= 0 and t2 <= 1:
            # ExitWound
            intersection_x = line_start[0] + t2*d[0]
            intersection_y = line_start[1] + t2*d[1]
            return (True, (intersection_x, intersection_y)) ;
        
        # No intersection: FallShort, Past, CompletelyInside
        # NOTE: maybe CompletelyInside should hit but then we wouldn't have an intersection point
        return (False, (-1, -1))
        
        # collides = False
        # intersection_pts = []
        # for internal_line in self.lines():
        #     line_collides, intersection_pt = Cell.test_collision(internal_line, line)
        #     if line_collides:
        #         collides = True
        #         intersection_pts.append(intersection_pt)
            
        # return (collides, intersection_pts)

    def collides_with_point(self, point):
        dist = math.dist(self.center, point)
        return dist <= self.collision_radius

    def draw(self, img, radius=None):
        if radius is None:
            radius = self.collision_radius
        cv.circle(img, self.center, radius, (0,0,255), 2)
