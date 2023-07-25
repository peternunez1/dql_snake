from turtle import Turtle
ALIGNMENT = "center"
FONT = ("Courier", 20, "normal")


class Scoreboard(Turtle):
    def __init__(self):
        super().__init__()
        self.current_score = 0
        self.color("white")
        self.penup()
        self.goto(0, 7)
        self.update_score()
        self.hideturtle()
        self.game_is_on = 1
        self.experience_counter = 0  # Use this variable globally

    def update_score(self):
        self.write(f"Score: {self.current_score}", align=ALIGNMENT, font=FONT)

    def increase_score(self):
        self.clear()
        self.update_score()

    def game_over(self):
        self.goto(0, 0)
        self.write("GAME OVER", align=ALIGNMENT, font=FONT)
        self.game_is_on = 0



