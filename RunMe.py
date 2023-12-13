import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义全局变量
Global = {}
Global['num_satellite'] = 4  # 地面站数量
Global['num_object'] = 12  # 观测目标数（卫星数）
Global['rank_satellite'] = None
Global['rank_object'] = None
Global['sat_need_time'] = None
Global['visible_window'] = {}
Global['num_visible_window'] = np.zeros((Global['num_object'], Global['num_satellite']))
# 读取数据
# 卫星优先级（列表/矩阵）
data_G = pd.read_csv('data/G.csv', encoding='ANSI').iloc[0:4, 1]
Global['rank_satellite'] = data_G.values
#print(Global['rank_satellite'])
# 观测目标优先级（列表/矩阵）
data_P = pd.read_csv('data/P.csv', encoding='ANSI').iloc[0:12, 1]
Global['rank_object'] = data_P.values
# 观测目标观测时长（列表/矩阵）
data_need = pd.read_csv('data/need.csv', encoding='ANSI').iloc[0:12, 1]
Global['sat_need_time'] = data_need.values

#读取卫星数据
for i in range(0, Global['num_object']):
    datfile = f'data/sat{i + 1}.csv'
    data = pd.read_csv(datfile, header=None).iloc[0:4, 1:13].values
    for j in range(0, Global['num_satellite']):
        index = data[j] != 0
        Global['visible_window'][(i, j)] = data[j][index]
        # print( Global['visible_window'][(i, j)])
        Global['num_visible_window'][i, j] = len(Global['visible_window'][(i, j)]) // 2
        #  print(Global['num_visible_window'][i, j])
print('---test---')


def Init(N):
    empty = {'decs': None, 'objs': None, 'cons': None}
    population = [empty.copy() for _ in range(N)]

    for i in range(N):
        decs = np.concatenate([np.random.permutation(Global['num_object']) + 1,#卫星随机排列
                               np.random.randint(1, Global['num_satellite'] + 1, Global['num_object'])])
        # print(decs)
        population[i]['decs'] = decs.tolist()
        # print(population[i]['decs'])
    # Assuming there is a CalObj function to calculate objectives for the population
    population = CalObj(population)
    return population


def CalObj(population):
    N = len(population)
    for i in range(N):
        ind = population[i]
        # print(ind)
        object_list = ind['decs'][:Global['num_object']]
        # print(object_list)
        satellite_list = ind['decs'][Global['num_object']:]
        satellite_next_release_time = np.zeros(Global['num_satellite'])
        time_start_guance = np.zeros(Global['num_object'])
        # print(time_start_guance)
        time_end_guance = np.zeros(Global['num_object'])
        index_window_guance = np.zeros(Global['num_object'])
        cons = 0

        for j in range(Global['num_object']):
            # print((i,j))
            cur_object = object_list[j] - 1
            cur_satellite = satellite_list[j] - 1
            flag = 0
            for m in range(1, int(Global['num_visible_window'][cur_object, cur_satellite]) + 1):
                # if m==5:
                #     os.system("pause")
                time_start = Global['visible_window'][(cur_object, cur_satellite)][2 * m - 2]
                time_end = Global['visible_window'][(cur_object, cur_satellite)][2 * m - 1]
                # print((time_start, time_end))
                if satellite_next_release_time[cur_satellite] > time_end:
                    continue
                time_begin = max(satellite_next_release_time[cur_satellite], time_start)
                if time_begin < time_end - Global['sat_need_time'][cur_object]:
                    time_start_guance[cur_object] = time_begin
                    time_end_guance[cur_object] = time_start_guance[cur_object] + Global['sat_need_time'][cur_object]
                    satellite_next_release_time[cur_satellite] = time_end_guance[cur_object] + 60
                    index_window_guance[cur_object] = m
                    flag = 1
                    break
            if flag == 0:
                cons += Global['sat_need_time'][cur_object]
        T = max(time_end_guance)
        total_rank = 0
        for j in range(Global['num_object']):
            cur_object = object_list[j] - 1
            cur_satellite = satellite_list[j] - 1
            total_rank += Global['rank_satellite'][cur_satellite] * Global['rank_object'][cur_object]

        population[i]['objs'] = T - 10 * total_rank
        population[i]['cons'] = cons
        population[i]['time_start_guance'] = time_start_guance
        population[i]['time_end_guance'] = time_end_guance
        population[i]['satellite_list'] = satellite_list
        print(population[i]['satellite_list'])
        # 观测窗口编号
        population[i]['index_window_guance'] = index_window_guance
        population[i]['satellite_next_release_time'] = satellite_next_release_time
    return population

#Mutate
def Mutate(population, state):
    # population = init(10)
    N = len(population)
    empty = {'decs': None, 'objs': None, 'cons': None}
    Offspring_temp = [empty.copy() for _ in range(N)]
    # Offspring = population.copy()
    for i in range(N):
        p1 = population[i]['decs'].copy()
        # print("交叉之前的offspring数据：\n",population[i]['decs'])
        if np.random.rand() < 0.8:  # 交叉
            # p2 = np.random.randint(N)
            while 1:
                p2_temp = np.random.randint(N)
                if p2_temp == i:
                    continue
                else:
                    break
                # p2 = np.random.randint(N)
            p2 = population[p2_temp]['decs'].copy()
            # 随机挑选两个进行片段交叉
            pos = np.sort(np.random.permutation(Global['num_object'])[:2])
            for j in range(pos[0], pos[1] + 1):
                if p1[j] != p2[j]:
                    ind = p1[:Global['num_object']].index(p2[j])
                    p1[ind] = p1[j]
                    p1[j] = p2[j]
            p1[pos[0] + Global['num_object']:pos[1] + Global['num_object'] + 1] = p2[pos[0] + Global['num_object']:pos[1] +Global['num_object'] + 1]

            if len(p1) < 24:
                print("测试交叉是否这里出现了问题：", len(p1), "第", i, "次")
                print("此时交叉的位置是:", pos)

        if np.random.rand() < 0.4:  # 变异
            pos = np.sort(np.random.permutation(Global['num_object'])[:2])  # 随机挑选两个位置进行片段逆转
            tmp = p1.copy()
            p1[pos[0]:pos[1] + 1] = np.flipud(tmp[pos[0]:pos[1] + 1])
            if len(p1) < 24:
                print("测试变异是否这里出现了问题：", len(p1), "第", i, "次")
                print("此时变异的位置是:", pos)
            pos = pos + Global['num_object']
            p1[pos[0]:pos[1] + 1] = np.flipud(tmp[pos[0]:pos[1] + 1])
        Offspring_temp[i]['decs'] = p1
    # for i_temp in range(N):
    #     Offspring[i_temp]['decs'][0:23] = Offspring_temp[i_temp]
    Offspring = CalObj(Offspring_temp)
    return Offspring

# 挑选新个体
def Select(population, offspring, N):
    # 对每一代种群中的染色体进行选择，以进行后面的交叉和变异
    joint = population + offspring
    objs = np.array([ind['objs'] for ind in joint])
    cons = np.array([ind['cons'] for ind in joint])
    index = np.lexsort((objs, cons))
    joint = [joint[i] for i in index]
    # 删除重复个体
    del_indices = []
    for i in range(len(joint) - 1):
        if i in del_indices:
            continue
        for j in range(i + 1, len(joint)):
            if joint[i]['decs'] == joint[j]['decs']:
                del_indices.append(j)
    joint = [joint[i] for i in range(len(joint)) if i not in del_indices]
    population = joint[:N]
    return population


# 主函数，整个函数从这里开始
if __name__ == '__main__':
    # 算法参数
    maxgen = 300
    popsize = 150
    population = Init(popsize)

    trace_obj = np.zeros(maxgen)
    trace_con = np.zeros(maxgen)
    # 进化开始
    for i in range(maxgen):
        # 交叉变异
        offspring = Mutate(population, i / maxgen)
        # 挑选新个体
        population = Select(population, offspring, popsize)
        # print(len(population))
        # 记录信息
        bestobj = population[0]['objs']
        trace_obj[i] = bestobj
        trace_con[i] = population[0]['cons']

        if not i % 10:
            cons = [ind['cons'] for ind in population]
            num = sum(1 for c in cons if c == 0)
            avgcons = np.mean(cons)
            print(f'第 {i} 代，满足约束个体数量：{num}，最佳个体：{bestobj}')

    # 进化结束
    print("---进化结束---\n---下面是绘图操作---")
    # 展示结果
    plt.plot(trace_obj) # 进化的迭代图
    # plt.title('最优目标值进化示意图')
    plt.show()

    bestsol = population[0]
    # Assuming that the necessary data and variables are available
    # Using 'jet' colormap to create a color map
    cmap = plt.get_cmap('jet')
    c_space = cmap(np.linspace(0, 1, Global['num_object']))
    color_list = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#10c020', '#d20F99', '#FF00FF',
                  '#00FFFF', '#10c020', '#d20F99']
    # print(c_space_temp[1][0:3])
    # c_space = np.delete(c_space_temp,0,3)
    # First part of the code
    for i in range(1, Global['num_object'] + 1):
        cur_object = bestsol['decs'][i - 1]
        cur_satellite = bestsol['satellite_list'][i - 1]
        # print(cur_satellite)
        ind_window = int(bestsol['index_window_guance'][cur_object - 1])

        t_s = bestsol['time_start_guance'][cur_object - 1]
        t_e = bestsol['time_end_guance'][cur_object - 1]
        t_s_window = Global['visible_window'][(cur_object - 1, cur_satellite - 1)][2 * ind_window - 2]
        t_e_window = Global['visible_window'][(cur_object - 1, cur_satellite - 1)][2 * ind_window - 1]
        if t_s == 0 and t_e == 0:
            continue

        ax = plt.subplot(4, 3, cur_object)
        rect1 = patches.Rectangle((t_s_window, cur_satellite - 0.1), t_e_window - t_s_window, 0.2, facecolor=c_space[i - 1],edgecolor='k',fill=True)
        ax.add_patch(rect1)
        rect2 = patches.Rectangle((t_s, cur_satellite - 0.25), t_e - t_s, 0.5, facecolor=c_space[i - 1],edgecolor='k',fill=True)
        ax.add_patch(rect2)
        plt.text(t_s + 100, cur_satellite, str(cur_object), fontweight='bold', fontsize=8)
        x_min = np.min([t_s_window,t_s])
        x_max = np.max([t_e_window,t_e])
        plt.ylim([0, 5])
        plt.xlim([x_min-500,x_max+500])
        plt.title('Object ' +  str(cur_object), fontsize=10)
    plt.show()

    ax = plt.figure(2)
    for i in range(1, Global['num_object'] + 1):
        cur_object = bestsol['decs'][i - 1]
        cur_satellite = bestsol['satellite_list'][i - 1]
        ind_window = int(bestsol['index_window_guance'][cur_object - 1])
        t_s = bestsol['time_start_guance'][cur_object - 1]
        t_e = bestsol['time_end_guance'][cur_object - 1]
        t_s_window = Global['visible_window'][(cur_object - 1, cur_satellite - 1)][2 * ind_window - 2]
        t_e_window = Global['visible_window'][(cur_object - 1, cur_satellite - 1)][2 * ind_window - 1]
        if t_s == 0 and t_e == 0:
            continue
        rect1 = patches.Rectangle((t_s_window, cur_satellite - 0.1), t_e_window - t_s_window, 0.2, facecolor=c_space[i - 1],edgecolor='k',fill=True)
        plt.gca().add_patch(rect1)
        rect2 = patches.Rectangle((t_s, cur_satellite - 0.25), t_e - t_s, 0.5, facecolor=c_space[i - 1],edgecolor='k',fill=True)
        plt.gca().add_patch(rect2)
        plt.text(t_s + 100, cur_satellite, str(cur_object), fontweight='bold', fontsize=8)
    for i in np.arange(0.5, 4.6, 1):
        plt.hlines(i, min(bestsol['time_start_guance']), max(bestsol['time_end_guance']), linestyles='-.')
    plt.show()

