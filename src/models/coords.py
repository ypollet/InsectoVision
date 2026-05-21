class Point:
    def __init__(self,x,y, itemId = None):
        self.x = x
        self.y = y
        self.itemId = itemId
    
    def move(self, x, y):
        dx = x - self.x
        dy = y - self.y
        self.x = x
        self.y = y

        return dx, dy
    
    def scale(self, width, height):
        return Point(self.x * width, self.y * height, self.itemId)
    
    def to_list(self):
        return [self.x,self.y]
    
    def __str__(self):
        return f"{self.x}x{self.y}"

class Coords:
    def __init__(self, first : Point, sec : Point):
        self.first = first
        self.second = sec

    def from_coords(x1,y1,x2,y2):
        return Coords(Point(x1, y1),Point(x2, y2))
    
    
    def scale(self, width, height):
        return Coords(self.first.scale(width, height), self.second.scale(width, height))
    
    def to_list(self):
        x1, y1 = self.first.to_list()
        x2, y2 = self.second.to_list()
        # left, top, right, bottom
        return [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
    
    def __str__(self):
        return f"{self.first} - {self.second}"