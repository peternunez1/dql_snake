from turtle import Screen, Turtle

STARTING_POSITIONS = [(0, 0), (-0.5, 0), (-1, 0)]
UP = 90
DOWN = 270
LEFT = 180
RIGHT = 0


class Snake(Turtle):
    def __init__(self):
        super().__init__()
        self.segments = []
        self.create_snake()
        self.head = self.segments[0]
        self.action_list = [self.turn_upward, self.turn_left, self.turn_downward, self.turn_right]

    def create_snake(self):
        for position in STARTING_POSITIONS:
            self.add_segment(position)

    def add_segment(self, position):
        new_segment = Turtle(shape="square")
        new_segment.color("green")
        new_segment.penup()
        new_segment.shapesize(1, 1)
        new_segment.goto(position)
        self.segments.append(new_segment)

    def extend(self):
        self.add_segment(self.segments[-1].position())

    def move(self):
        for i in range(len(self.segments) - 1, 0, -1):
            new_x = self.segments[i - 1].xcor()
            new_y = self.segments[i - 1].ycor()
            self.segments[i].goto(new_x, new_y)
        self.head.forward(0.5)

    def turn_upward(self):
        if self.head.heading() != DOWN:
            self.head.setheading(UP)

    def turn_left(self):
        if self.head.heading() != RIGHT:
            self.head.setheading(LEFT)

    def turn_downward(self):
        if self.head.heading() != UP:
            self.head.setheading(DOWN)

    def turn_right(self):
        if self.head.heading() != LEFT:
            self.head.setheading(RIGHT)

    def clear_segments(self):
        for segment in self.segments:
            segment.goto(100, 100)  # Move segments off the screen
        self.segments.clear()

