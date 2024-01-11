class City:
    def __init__(self, node):
        self.node = node
        self.links = []

    def add_link(self, link):
        self.links.append(link)

    def draw(self, img):
        self.node.draw_hitbox(img, radius=12)
        