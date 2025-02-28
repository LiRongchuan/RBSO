import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

EPISILON = 1E-10

class Environment:
    
    def __init__(self, area=(20, 20), num_obstacles=10, num_targets=5, num_robots=5, a=1, r=2, threshold=0.5, seed=666):
        self.area = area
        self.num_obstacles = num_obstacles
        self.num_targets = num_targets
        self.num_robots = num_robots
        self.a = a
        self.r = r
        self.threshold = threshold
        # np.random.seed(seed)
        self.obstacles = self.generate_positions(num_obstacles, area)
        self.targets = self.generate_positions(num_targets, area)
        self.detected = np.full(num_targets, False)
        self.robots = self.generate_positions(num_robots, area)
        self.signals = np.zeros(num_robots)
        self.storage = [[] for _ in range(num_robots)]
        self.surrounding_obstacles = [[] for _ in range(num_robots)]
        self.surrounding_robots = [[] for _ in range(num_robots)]
        
    def generate_positions(self, num, area):
        x = np.random.rand(num) * area[0]
        y = np.random.rand(num) * area[1]
        return np.column_stack((x, y))
    
    def signal_strength(self, d):
        return 1 / (self.a * np.exp(d**2) + EPISILON)
    
    def measure(self):
        for i, robot in enumerate(self.robots):
            for j, target in enumerate(self.targets):
                if not self.detected[j]:
                    d = np.linalg.norm(robot - target)
                    strength = self.signal_strength(d)
                    self.signals[i] += strength
                    if strength >= self.threshold:
                        self.detected[j] = True
            self.storage[i].append([robot[0], robot[1], self.signals[i]])
        return self.signals
    
    def perceive(self):
        for i, robot in enumerate(self.robots):
            surrounding_obstacles = []
            surrounding_robots = []
            for obstacle in self.obstacles:
                if np.linalg.norm(robot - obstacle) <= self.r:
                    surrounding_obstacles.append([obstacle[0], obstacle[1]])
            for j, neighbor in enumerate(self.robots):
                if np.linalg.norm(robot - neighbor) <= self.r and j != i:
                    surrounding_robots.append([j, neighbor[0], neighbor[1]])
            self.surrounding_obstacles[i] = surrounding_obstacles
            self.surrounding_robots[i] = surrounding_robots
        return self.surrounding_obstacles, self.surrounding_robots
        
    def plot(self):
        grid_size = 100
        x = np.linspace(0, self.area[0], grid_size)
        y = np.linspace(0, self.area[1], grid_size)
        xv, yv = np.meshgrid(x, y)

        signal_grid = np.zeros_like(xv)
        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([xv[i, j], yv[i, j]])
                for k, target in enumerate(self.targets):
                    if not self.detected[k]:
                        d = np.linalg.norm(point - target)
                        signal_grid[i, j] += self.signal_strength(d)
                    
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.area[0])
        ax.set_ylim(0, self.area[1])
        
        ax.contourf(xv, yv, signal_grid, levels=20, cmap='viridis', alpha=0.6)
        ax.scatter(self.obstacles[:, 0], self.obstacles[:, 1], c='black', marker='^', label='Obstacles')
        ax.scatter(self.targets[self.detected][:, 0], self.targets[self.detected][:, 1], c='green', marker='*', label='Detected Targets')
        ax.scatter(self.targets[~self.detected][:, 0], self.targets[~self.detected][:, 1], c='red', marker='*', label='Undetected Targets')
        ax.scatter(self.robots[:, 0], self.robots[:, 1], c='blue', marker='s', label='Robot')
        
        for i, robot in enumerate(self.robots):
            ax.text(robot[0], robot[1], f'{self.signals[i]:.2f}', fontsize=9, ha='right')
        
        plt.legend()
        plt.grid()
        plt.show()
        
if __name__ == '__main__':
    env = Environment()
    env.measure()
    env.plot()