from turtle import Turtle

ALIGNMENT = "left"
FONT = ("Courier", 8, "normal")


class Epoch(Turtle):
    def __init__(self):
        super().__init__()
        self.current_epoch = 1
        self.color("white")
        self.penup()
        self.goto(-6, -7)
        self.hideturtle()
        self.update_epoch()

    def update_epoch(self):
        self.write(f"Epoch: {self.current_epoch}", align=ALIGNMENT, font=FONT)

    def increase_epoch(self):
        self.clear()
        self.current_epoch += 1
        self.update_epoch()
        print("New Epoch...")

