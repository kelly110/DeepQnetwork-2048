from tkinter import *
from logic import *
from random import *
import time

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")
"""
KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"
"""
class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        #self.master.bind("<Key>", self.key_b_press)

        #self.gamelogic = gamelogic
        """self.commands = {   KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right,
                            KEY_UP_ALT: up, KEY_DOWN_ALT: down, KEY_LEFT_ALT: left, KEY_RIGHT_ALT: right }
        """
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.n_features = 11

        self.grid_cells = []
        self.init_grid()
        #self.init_matrix()
        #self.update_grid_cells()
        
        #self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

        """def key_b_press(self,event):
        key = repr(event.char)
        return key
        print("key_b_press",key)
        if key=='b':
            self.close()"""



    def init_matrix(self):
        self.update()

        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)
        s=new_s(self.matrix)
        # s=unification(s)
        #print("puzzle init_matrix")
        # print(s)
        return s

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        #print("puzzle update_grid_cells")
        self.update_idletasks()#call update
        #print("puzzle end_update_grid_cells")

        
    """def key_down(self, event):
        key = repr(event.char)
        if key in self.commands:
            self.matrix,done = self.commands[repr(event.char)](self.matrix)
            if done:
                self.matrix = add_two(self.matrix)
                self.update_grid_cells()
                done=False
                if game_state(self.matrix)=='win':
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!",bg=BACKGROUND_COLOR_CELL_EMPTY)
                if game_state(self.matrix)=='lose':
                    self.grid_cells[1][1].configure(text="You",bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!",bg=BACKGROUND_COLOR_CELL_EMPTY)
    """



    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2



    def step(self, action):
        #print("puzzle step")

        #s = self.matrix
        #print("puzzle step s",self.matrix)
        if action == 0:   # up
            self.matrix, done, reward=up(self.matrix)
        elif action == 1:   # down
            self.matrix, done, reward=down(self.matrix)
        elif action == 2:   # right
            self.matrix, done, reward =right(self.matrix)
        elif action == 3:   # left
            self.matrix, done, reward =left(self.matrix)
        max_sum=max_mat(self.matrix)
        if(reward==0):
            reward=-2
        #print("puzzle before done s_, done,reward", self.matrix, done, reward)
        is_end=False
        is_win=0
        if done:
            self.matrix = add_two(self.matrix)
            #****************************************************************************************
            self.update_grid_cells()
            done = False
            if game_state(self.matrix) == 'win':
                self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(text="Win!", bg=BACKGROUND_COLOR_CELL_EMPTY)
                self.update_grid_cells()
                is_end=True
                is_win=1
                print("\nyou win\n")
                time.sleep(3)
            if game_state(self.matrix) == 'lose':
                self.grid_cells[1][1].configure(text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)
                self.update_grid_cells()
                reward=-50
                is_end = True
                print("\nyou lose\n")
                time.sleep(1)
        s_ = new_s(self.matrix)
        # s_ = unification(s_)
        # print("puzzle had done s_, done,reward",s_, done,reward)

        return (s_, done,reward,is_end,is_win)

    def render(self):
        # time.sleep(0.01)
        self.update()

#gamegrid = GameGrid()
