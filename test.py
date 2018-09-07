from RL_brain import PolicyGradient
from emulator import *
NUM = 12


def get_inputs(data):
    # env.map, env.gas_pos, env.weapon_pos, player_a, player_b, path_array
    obstacle_map = data['map']
    gas_map = data['gas_map']
    props_map = np.zeros(shape=(NUM, NUM))
    player_a = np.zeros(shape=(NUM, NUM))
    player_b = np.zeros(shape=(NUM, NUM))
    path = np.zeros(shape=(NUM, NUM))
    cur_player = data['cur_player']

    # set players and path
    for player, pos in data['players'].items():
        if player == cur_player:
            player_a[pos[0], pos[1]] = 1
            path[pos[0], pos[1]] = 1
        else:
            player_b[pos[0], pos[1]] = 1

    # set props_map
    for prop, pos in data["props"].items():
        props_map[pos[0], pos[1]] = 1

    mats = np.array([obstacle_map, gas_map, props_map, player_a, player_b, path]).transpose((1, 2, 0))
    return mats


def get_valid_action(map_, row, col, con_row, con_col):
    actions = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if 0 <= row + i < 12 and 0 <= col + j < 12:
                if map_[row + i][col + j] > 0 or (row + i == con_row and col + j == con_col):
                    actions.append(0)
                else:
                    actions.append(1)
            else:
                actions.append(0)
    return actions


def next(state, action):
    """

    :param state: 12*12*6 :array
    :param action: 0-9
    :return: next_state:12*12*6, next_player_pos:1*8
    """
    # input 12*12*5
    # env.map, env.gas_pos, env.weapon_pos, player_a, player_b, path_array
    next_state = None
    map_ = state[:, :, 0]
    gas_pos = state[:, :, 1]
    weapon = state[:, :, 2]
    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
    path = state[:, :, -1]
    in_gas = False

    if gas_pos[player_pos[0]][player_pos[1]] > 0:
        in_gas = True
    r = new_r = int(player_pos[0])
    c = new_c = int(player_pos[1])
    if action == 0 or action == 4 or action == 5:  # up
        new_r -= 1
    if action == 1 or action == 6 or action == 7:  # down
        new_r += 1
    if action == 2 or action == 4 or action == 6:  # left
        new_c -= 1
    if action == 3 or action == 5 or action == 7:  # right
        new_c += 1
    is_valid = True
    has_weapon = False
    has_repetition = False
    if new_r < 0 or new_c < 0 or new_r >= 12 or new_c >= 12:
        is_valid = False
    else:
        if map_[new_r, new_c] > 0:
            is_valid = False
        elif new_r == enemy_pos[0] and new_c == enemy_pos[1]:
            is_valid = False
        else:
            if weapon[new_r, new_c] > 0:
                has_weapon = True
                weapon[new_r, new_c] = 0
            if path[new_r, new_c] > 0:
                has_repetition = True
            path[new_r, new_c] = 1
            state[:, :, 3][r, c] = 0
            state[:, :, 3][new_r, new_c] = 1
    #valid_action = (get_valid_action(map_, new_r, new_c, enemy_pos[0], enemy_pos[1])
    #                if is_valid else [0]*8)
    # modify by tj
    valid_action = get_valid_action(map_, new_r, new_c, enemy_pos[0], enemy_pos[1])
    has_attack = is_attack(map_, enemy_pos, path)

    return is_valid, state, valid_action, has_weapon, in_gas, has_attack, has_repetition


def is_adjacent(row_1, col_1, row_2, col_2):
    r_dist = row_1 - row_2
    c_dist = col_1 - col_2
    if abs(r_dist) + abs(c_dist) == 1:
        return True
    return False


def get_adj_four(map_, r_pos, c_pos):
    adj = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) + abs(j) != 1:
                continue
            new_r = r_pos + i
            new_c = c_pos + i
            if new_r < 0 or new_c < 0 or new_r >= map_.shape[0] or new_c >= map_.shape[1]:
                continue
            if map_[new_r][new_c] == 0:
                adj.append((new_r, new_c))
    return adj


def is_attack(map_, enemy, path):
    adj = get_adj_four(map_, enemy[0], enemy[1])
    for k in adj:
        if path[k[0]][k[1]] > 0:
            return True
    return False


def get_reward(state, valid, action, has_weapon, in_gas, has_attack, has_repetition):
    map_ = state[:, :, 0]
    gas_pos = state[:, :, 1]
    player_pos = np.argwhere(state[:, :, 3] == 1)[0]
    enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
    path = state[:, :, -1]
    r = 0

    #hit wall or hit enemy is invalid
    if not valid:
        return -1.0
    elif action == 8 and len(path[path > 0]) == 1:
        return -1.0
    if has_repetition:
    #elif path[player_pos[0]][player_pos[1]] > 0:
        r -= 1
    if not in_gas and (gas_pos[player_pos[0]][player_pos[1]] > 0).any():
        r -= 0.5

    if is_adjacent(player_pos[0], player_pos[1], enemy_pos[0], enemy_pos[1]):
        #r += 0.3
        r += 0.9
    if has_weapon:
        #r += 0.15
        r += 0.5
    if action == 8 and not has_attack:
        r -= 1
    elif action == 8 and has_attack:
        adj = get_adj_four(map_, player_pos[0], player_pos[1])
        r += (4 - len(adj)) * 0.2
    return r


def run_game():
    step = 0
    for _ in range(1000000):
        eml = Emulator(p=0.1)  # 重置模拟器
        while 1:
            s = eml.get_state()
            if s['is_done']:  # 如果进入终盘则跳出
                break
            state = get_inputs(s)  # 将模拟器返回状态转换为12*12*6 tensor
            map_ = state[:, :, 0]
            player_pos = np.argwhere(state[:, :, 3] == 1)[0]
            enemy_pos = np.argwhere(state[:, :, 4] == 1)[0]
            actions = np.array(get_valid_action(map_,
                                                int(player_pos[0]), int(player_pos[1]),
                                                enemy_pos[0], enemy_pos[1]))

            observation = np.hstack((state.flatten(), actions))  # 展平并拼接

            moves = [player_pos]  # 路径列表
            #has_attack = False
            while True:
                # RL choose action based on observation
                action = RL.choose_action(observation=observation)
                # RL take action and get next observation and reward
                valid, state, actions, has_weapon, in_gas, has_attack, has_repetition = next(state, action)  # 状态转移
                observation_ = np.hstack((state.flatten(), actions))
                done = 0 if action != 8 and valid else 1  # 本轮是否终止
                reward = get_reward(state, valid, action, has_weapon, in_gas, has_attack, has_repetition)
                RL.store_transition(observation_, action, reward)
                player_pos = np.argwhere(state[:, :, 3] == 1)[0]
                moves.append(player_pos)
                if done:
                    RL.learn()
                    step += 1
                    #if step % 1000 == 0:
                    #    print('step:' + str(step))
                    break
                # swap observation
                observation = observation_
            #eml.update_player_pos(player_pos)
            eml.next(list(map(lambda x: (x[0], x[1]), moves)))


if __name__ == "__main__":
    RL = PolicyGradient(n_actions=9, n_features=144 * 6 + 8, learning_rate=1e-8)
    run_game()
