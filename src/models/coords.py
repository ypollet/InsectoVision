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
        return self.first.to_list() + self.second.to_list()
    
    def __str__(self):
        return f"{self.first} - {self.second}"