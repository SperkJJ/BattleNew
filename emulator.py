import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random


class Emulator(object):
    __COLOR_MAP = colors.ListedColormap(['#FFCC99', '#FF9966', '#99CCCC', '#0099CC', '#FF6666', '#CCCCCC', '#666666'])
    __NORM = colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7], __COLOR_MAP.N)

    @staticmethod
    def map_initializer(p):
        return np.random.choice(2, size=(12, 12), p=(1 - p, p)).astype(np.int32)

    def __init__(self, map_=None, p=0.15, round_=1, players=None):
        if not map_:
            map_ = self.map_initializer(p)
        self.map = map_
        if not players:
            players = self.generate_players()
        self.players = players
        self.props = {}
        self.cur_player = 0
        self.round = round_
        self.count = 0
        self.gas = np.zeros((12, 12), dtype=np.int32)
        self.__update_gas()
        self.__update_props()
        self.is_done = False

    def display(self):
        graph = self.map.copy()
        graph[(self.gas != 0) & (self.map == 0)] = 2
        for player_name, player_pos in self.players.items():
            if player_name == 0:
                graph[player_pos[0], player_pos[1]] = 3
            if player_name == 1:
                graph[player_pos[0], player_pos[1]] = 4

        for prop_name, prop_pos in self.props.items():
            if prop_name == 0:
                graph[prop_pos[0], prop_pos[1]] = 5
            if prop_name == 1:
                graph[prop_pos[0], prop_pos[1]] = 6

        plt.imshow(graph, cmap=self.__COLOR_MAP, norm=self.__NORM)
        plt.title('round {self.round}, player{self.cur_player}\'s turn')
        plt.show()

    def generate_players(self):
        players = {0: (0, 0), 1: (11, 11)}
        for i in range(2):
            while True:
                r_pos = random.randint(0, 11)
                c_pos = random.randint(0, 11)
                if self.map[r_pos][c_pos] == 0 and self.is_valid_wall(r_pos, c_pos) and (players[1 - i][0] != r_pos or players[1 - i][1] != c_pos):
                    players[i] = (r_pos, c_pos)
                    break
        return players

    def __update_gas(self):
        radius = 6 - self.round // 5
        self.gas[:, :] = 1
        self.gas[6 - radius: 6 + radius, 6 - radius: 6 + radius] = 0

    def __update_props(self):
        if self.round % 5 != 1:
            self.props = {}
            return
        self.props = {0: (0, 0), 1: (0, 0)}
        for i in range(2):
            while True:
                r_pos = random.randint(0, 11)
                c_pos = random.randint(0, 11)
                if self.map[r_pos][c_pos] == 0 and \
                        self.is_valid_wall(r_pos, c_pos) and \
                        (self.players[self.cur_player][0] != r_pos or self.players[self.cur_player][1] != c_pos) and \
                        (self.players[1 - self.cur_player][0] != r_pos or self.players[1 - self.cur_player][
                            1] != c_pos) and \
                        (self.props[1 - i][0] != r_pos or self.props[1 - i][1] != c_pos):
                    self.props[i] = (r_pos, c_pos)
                    break

    def __update(self):
        # update count
        self.count += 1

        # update round
        if self.count % 2 == 0:
            self.round += 1

        if self.round == 30 and self.count % 2:
            self.is_done = True

        # update gas_map
        self.__update_gas()

        # update player status
        self.cur_player = 1 - self.cur_player

        # update props status
        self.__update_props()

    def update_player_pos(self, player_pos):
        self.players[self.cur_player] = player_pos

    def get_state(self):
        return {"map": self.map, "gas_map": self.gas, "props": self.props, "players": self.players,
                "cur_player": self.cur_player, "round": self.round, "is_done": self.is_done}

    def next(self, moves):
        # init end_step and enemy_pos
        enemy_pos = self.players[1-self.cur_player]

        path = set()
        is_break = False
        last_step = self.players[self.cur_player]

        # print(last_step, enemy_pos)

        for step in moves:    # moves = {(0,1),(1,2),...}
            # judge step out of board
            if step[0] < 0 or step[1] < 0 or step[0] >= 12 or step[1] >= 12:
                #print("step out of board")
                is_break = True
                break

            # judge step coincide with walls
            if self.map[step[0]][step[1]] == 1:
                #print("step coincide with walls")
                is_break = True
                break

            # judge step repeat
            if step in path:
                #print("step repeat")
                is_break = True
                break

            # judge step coincide with enemy
            if step[0] == enemy_pos[0] and step[1] == enemy_pos[1]:
                #print("step coincide with enemy")
                is_break = True
                break

            # judge step not continuous
            d = (step[0]-last_step[0], step[1]-last_step[1])
            if abs(d[0]) > 1 and abs(d[1]) > 1:
                #print("step not continuous")
                is_break = True
                break

            path.add(step)
            last_step = step

        # record end_step
        #if not is_break:
        self.players[self.cur_player] = (moves[-1][0], moves[-1][1])

        # update state
        self.__update()

    def is_valid_wall(self, r_pos, c_pos):
        adj = self.get_adj(r_pos, c_pos)
        return len(adj) >= 2

    def get_adj(self, r_pos, c_pos):
        adj = []
        for i in range(-1, 2):
            new_r = r_pos + i
            if new_r < 0 or new_r >= 12:
                continue
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                new_c = c_pos + j
                if new_c < 0 or new_c >= 12:
                    continue
                if self.map[new_r][new_c] == 0:
                    adj.append((new_r, new_c))
        return adj


if __name__ == '__main__':
    emulator = Emulator(p=0.1, round_=1)
    emulator.display()
