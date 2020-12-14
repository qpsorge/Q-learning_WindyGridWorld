import numpy as np 
import random

class Grid :
    ACTIONS_NAMES = ['UP','LEFT','DOWN','RIGHT']
    ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT = 0, 1, 2, 3
    ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT]
    
    MOVEMENTS = {
        ACTION_UP:    (-1 , 0),
        ACTION_RIGHT: ( 0 ,  1),
        ACTION_LEFT:  ( 0 , -1),
        ACTION_DOWN:  ( 1 ,  0)
    }

    AGENT = 5
    HOLE  = 1
    BLOCK = 8
    END   = 3

    def __init__(self, size_x=4, size_y=4, wrong_action_p=0):
        self.size_x, self.size_y = size_x, size_y
        self.wrong_action_p = wrong_action_p
        self.generate_game()
    def generate_game(self):
        self.step  = 0
        self.grid  = np.zeros((self.size_x,self.size_y))
        self.start = self.attribute_random_pos(self.AGENT) #S
        self.hole  = self.attribute_random_pos(self.HOLE) #H
        self.block = self.attribute_random_pos(self.BLOCK) #B
        self.end   = self.attribute_random_pos(self.END) #E
        self.agent = self.start
        return self._get_state()

    def attribute_random_pos(self, nb):
        x_coord = random.randint(0,self.size_x-1)
        y_coord = random.randint(0,self.size_y-1)
        while self.grid[x_coord, y_coord]!=0:
            x_coord = random.randint(0,self.size_x-1)
            y_coord = random.randint(0,self.size_y-1)
        self.grid[x_coord, y_coord]=nb
        return (x_coord, y_coord)

    def reset(self):
        self.step  = 0
        self.agent = self.start
        self.grid[self.start] =self.AGENT
        self.grid[self.hole]  =self.HOLE
        self.grid[self.block] =self.BLOCK
        self.grid[self.end]   =self.END
        return self._get_state()
    


    def get_agent_coord(self):
        return tuple([x[0] for x in np.where(self.grid==self.AGENT)])
    def get_block_coord(self):
        return tuple([x[0] for x in np.where(self.grid==self.BLOCK)])
    def get_hole_coord(self):
        return tuple([x[0] for x in np.where(self.grid==self.HOLE)])
    def get_end_coord(self):
        return tuple([x[0] for x in np.where(self.grid==self.END)])
    def _get_state(self):
        return self._position_to_id(self.get_agent_coord())

    def _position_to_id(self, coords):
        """Gives the position id (from 0 to n)"""
        x, y = coords[0], coords[1]
        return y + x * self.size_y
    def _id_to_position(self, id):
        """Réciproque de la fonction précédente"""
        return (id // self.size_y, id % self.size_y)

    def move(self, action):
        """
        takes an action parameter
        :param action : the id of an action
        :return ((state_id, end, hole, block), reward, is_final, actions)
        """
        self.step+=1
        if action not in self.ACTIONS:
            raise Exception('Invalid action')
        
        choice = random.random()
        if choice < self.wrong_action_p :
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.get_agent_coord()
        new_x, new_y = x + d_x, y + d_y

        if   self.get_block_coord() == (new_x, new_y): #If block we do not moove
            return self._get_state(), -1, False, self.ACTIONS
        elif self.get_hole_coord() == (new_x, new_y):
            self.update_pos_agent(x,y, new_x,new_y)
            return self._get_state(), -10, True, None

        elif self.get_end_coord() == (new_x, new_y):
            self.update_pos_agent(x,y, new_x,new_y)
            return self._get_state(), 10, True, self.ACTIONS

        elif new_x >= self.size_x or new_y >= self.size_y or new_x < 0 or new_y < 0:
            return self._get_state(), -1, False, self.ACTIONS

        elif self.step > 190:
            self.update_pos_agent(x,y, new_x,new_y)
            return self._get_state(), -10, True, self.ACTIONS

        else:
            self.update_pos_agent(x,y, new_x,new_y)
            return self._get_state(), -1, False, self.ACTIONS
        
    def update_pos_agent(self,x,y, new_x,new_y):
        # ADD WIND EFFECT
        # TO DO 
        self.agent = new_x, new_y                            # store position
        self.grid[x,y], self.grid[new_x,new_y]=0, self.AGENT # move agent in grid

    def __str__(self):
        sentence = "\nWINDY GRID WORLD \n"
        sentence += str(self.grid) +"\n\n"
        sentence += f"Agent ({self.AGENT}) coords : {self.get_agent_coord()}\n"
        sentence += f"Block ({self.BLOCK}) coords : {self.get_block_coord()}\n"
        sentence += f"Hole  ({self.HOLE}) coords : {self.get_hole_coord()}\n"
        sentence += f"End   ({self.END}) coords : {self.get_end_coord()}\n"
        return sentence

#g = Grid()
#print(g)
#print(g._position_to_id(g.get_agent_coord()))
#print(g._id_to_position(g._position_to_id(g.get_agent_coord())))

