import random

class World:
    def __init__(self):
        # self.goal_x = 1
        # self.goal_y = 10
        self.agent_x = False
        # self.agent_y = 0
    #
    # def found_goal(self):
    #     if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
    #         return True
    #     else:
    #         return False

    def evaluate(self):
        return self.agent_x

    def get_data(self):
        return (self.agent_x)

    def set_data(self, val):
        self.agent_x = val

    def live(self):
        rnd = random.random()
        if rnd < 0.5:
            self.state = False
        elif:
            self.state = True



class Agent:
    def __init__(self, world):
        self.world = world

    # possibilities to act
    def move_right(self):
        self.world.agent_x += 1

    def move_left(self):
        self.world.agent_x -= 1
    #
    # def move_up(self):
    #     self.world.agent_y += 1
    #
    # def move_down(self):
    #     self.world.agent_y -= 1
    #
    # # not used anymore
    # def random_move(self):
    #     rnd = random.random()
    #     if rnd < 0.25:
    #         self.move_right()
    #     elif rnd < 0.5:
    #         self.move_left()
    #     elif rnd <0.75:
    #         self.move_up()
    #     else:
    #         self.move_down()
    #     print(rnd, self.x_pos, self.y_pos)

    def live(self):
        step = 0
        while(not self.world.evaluate()):
            step += 1
            # data in the beginning is 1,0
            data = self.world.get_data()
            self.world.set_data(self.mutate(data))
            print(step)

    def mutate(self, data):
        rnd = random.random()
        if rnd < .1:
            return False
        elif rnd > .9:
            return True
        return data





world = World()
agent = Agent(world)
agent.live()