import os
os.environ["OMP_NUM_THREADS"] = '1'
import math
import numpy as np
from sklearn.cluster import KMeans

def logsig(x):
    return 1 / (1 + np.exp(-x))

def bso(func, population_size, dimension, cluster_number, left_range=-100, right_range=100, max_iteration=1000, seed=666):
    np.random.seed(seed)
    
    # 超参数
    center_mutate_probablity = 0.05 # 簇中心变异概率
    single_probability = 0.5 # 单簇变异概率
    singel_keep_probablity = 0.6 # 单簇中心稳定概率
    double_keep_probablity = 0.6 # 双簇中心结合概率

    # 变量定义
    population = left_range + (right_range - left_range) * np.random.rand(population_size, dimension) # 初始种群
    sorted_population = left_range + (right_range - left_range) * np.random.rand(population_size, dimension) #种群根据簇排序
    population_fitness = np.full(population_size, np.inf) # 每个个体适应度
    cluster = [0] * population_size # 每个个体所属簇
    sorted_population_fitness = np.full(population_size, np.inf) # 按所属簇排序
    centers = left_range + (right_range - left_range) * np.random.rand(cluster_number, dimension) # 初始簇中心
    accumulate_probablity = np.zeros(cluster_number) # 用于确保所有簇变异机会均等
    cluster_best = np.zeros(cluster_number, dtype=np.int32) # 每个簇中最优个体下标
    best_fitness = np.full(max_iteration, np.inf) # 每轮循环后最高适应度
    stepSize = np.ones(dimension) # 生成新个体的扰动范围
    mutate_individual = np.zeros(dimension) # 变异产生的新个体

    iteration_number = 0

    # 初始化种群
    for i in range(population_size):
        population_fitness[i] = func(population[i])

    # 迭代过程
    while iteration_number < max_iteration:

        # 对个体 k-means，生成簇
        kmeans = KMeans(n_clusters=cluster_number, max_iter=1000)
        kmeans.fit(population)
        cluster = kmeans.labels_.astype(int)

        cluster_fitness = np.full(cluster_number, np.inf)  # 初始化每个簇的最优适应度
        cluster_size = np.zeros(cluster_number, dtype=np.int32)  # 每个簇初始有0个个体
        # 找每个簇最优个体
        for i in range(population_size):
            cluster_size[cluster[i]] += 1
            # 更新最优适应度及个体（簇中心）
            if cluster_fitness[cluster[i]] > population_fitness[i]:
                cluster_fitness[cluster[i]] = population_fitness[i]
                cluster_best[cluster[i]] = i
                centers[cluster[i]] = population[i]

        # 根据所属簇重新排列个体
        individual_cnt = np.zeros(cluster_number, dtype=np.int32) # 本簇记录了多少个体
        cluster_cnt = np.zeros(cluster_number, dtype=np.int32) # 第i个簇前共多少个体
        for i in range(1, cluster_number):
            cluster_cnt[i] = cluster_cnt[i - 1] + cluster_size[i - 1]    
        for i in range(population_size):
            sorted_i = cluster_cnt[cluster[i]] + individual_cnt[cluster[i]]
            sorted_population[sorted_i] = population[i]
            sorted_population_fitness[sorted_i] = population_fitness[i]
            individual_cnt[cluster[i]] += 1

        # 中心变异
        if np.random.rand() < center_mutate_probablity:
            mutate_cluster_i = math.floor(np.random.rand() * cluster_number)
            centers[mutate_cluster_i] = left_range + (right_range - left_range) * np.random.rand()

        # 计算可能性
        accumulate_probablity[0] = cluster_size[0] / population_size
        for i in range(1, cluster_number):
            accumulate_probablity[i] = cluster_size[i] / population_size + accumulate_probablity[i - 1]
        
        # 高斯（正态）扰动产生新个体 mutate_individual，每个个体都有可能被一个不相关的新个体取代
        for i in range(population_size):
            mutate_type = np.random.rand()
            # 单簇变异
            if mutate_type < single_probability:
                cluster_mutate = np.random.rand()
                for j in range(cluster_number):
                    if cluster_mutate < accumulate_probablity[j]:
                        # 保持原中心
                        if np.random.rand() < singel_keep_probablity:
                            mutate_individual = centers[j]
                        # 随机替换一个中心
                        else:
                            parent_1 = cluster_cnt[j] + math.floor(np.random.rand() * cluster_size[j])
                            mutate_individual = sorted_population[parent_1]
                        break
            # 双簇变异
            else:
                cluster_1 = math.floor(np.random.rand() * cluster_number)
                cluster_2 = math.floor(np.random.rand() * cluster_number)
                split = np.random.rand()
                # 中心结合
                if np.random.rand() < double_keep_probablity:
                    mutate_individual = split * centers[cluster_1] + (1 - split) * centers[cluster_2]
                # 随机个体结合
                else:
                    parent_1 = cluster_cnt[cluster_1] + math.floor(np.random.rand() * cluster_size[cluster_1])
                    parent_2 = cluster_cnt[cluster_2] + math.floor(np.random.rand() * cluster_size[cluster_2])
                    if parent_1 == population_size:
                        parent_1 -= 1
                    if parent_2 == population_size:
                        parent_2 -= 1
                    mutate_individual = split * sorted_population[parent_1] + (1 - split) * sorted_population[parent_2]
            
            # 对新个体添加随机扰动
            stepSize = logsig(((0.5 * max_iteration - iteration_number) / 20)) * np.random.rand(dimension)
            mutate_individual += stepSize * np.random.normal(loc=0.0, scale=1.0, size=dimension)

            # 适应度有改进则更新种群
            new_fitness = func(mutate_individual)
            if new_fitness < population_fitness[i]:
                population_fitness[i] = new_fitness
                population[i] = mutate_individual
                
        # 确保最优点不变
        for i in range(cluster_number):
            population_fitness[cluster_best[i]] = cluster_fitness[i]
        
        # 记录迭代结果
        best_fitness[iteration_number] = min(cluster_fitness)
        iteration_number += 1
        
    return best_fitness