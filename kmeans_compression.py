from PIL import Image
import matplotlib as matplot
from matplotlib import pyplot as plt
import numpy as np
import random as rand

def dist(color1, color2, position1, position2):
    # print(color1, color1, position1, position2)
    return ((color1[0] - color2[0]) ** 2 + (color1[1] - color2[1]) ** 2 + (color1[2] - color2[2]) ** 2 + ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2))
    # return (color1[0] - color2[0]) ** 2 + (color1[1] - color2[1]) ** 2 + (color1[2] - color2[2]) ** 2

class ClusterManager:
    def __init__(self, num_clusters, image):
        self.num_clusters = num_clusters
        self.image = image
        self.init_clusters()

    def iterate(self):
        self.reset_clusters()
        self.assign_points()
        self.update_clusters()

    def init_clusters(self):
        cluster_list = []
        for i in range(self.num_clusters):
            position = (rand.randrange(0, self.image.shape[0]), rand.randrange(0, self.image.shape[1]))
            color = self.image[position[0]][position[1]]
            cluster_list.append(Cluster(self.image.shape, color, position))
        self.clusters = cluster_list

    def reset_clusters(self):
        for cluster in self.clusters:
            cluster.colors = []
            cluster.positions = []

    def assign_points(self):
        for i, row in enumerate(self.image):
            for k, pixel in enumerate(row):
                best_dist = float('inf')
                best_centroid = None
                for centroid in self.clusters:
                    distance = dist(centroid.color, pixel, centroid.position, (i, k))
                    if distance < best_dist: # that's a lot of nested loops... the leetcode part of my brain is cringing
                        best_centroid = centroid
                        best_dist = distance
                best_centroid.colors.append(pixel)
                best_centroid.positions.append([i, k])

    def update_clusters(self):
        self.clusters = [cluster for cluster in self.clusters if cluster.colors]
        for cluster in self.clusters:
            cluster.update_values()

    def apply_compression(self):
        new_image = np.zeros(self.image.shape)
        for i in range(self.image.shape[0]):
            for k in range(self.image.shape[1]):
                best_dist = float('inf')
                best_centroid = None
                for centroid in self.clusters:
                    distance = dist(self.image[i][k], centroid.color, (i, k), centroid.position)
                    if distance < best_dist:
                        best_centroid = centroid
                        best_dist = distance
                new_image[i][k] = np.array(best_centroid.color)
                # new_image[i][k] = np.array([rand.randrange(0, 255) for i in range(3)])
        return new_image
    
    def save_clusters(self, save_index):
        with open("./wolf_images/label{}".format(save_index), "w") as save_file:
            save_file.write("{},{}\n".format(self.image.shape[1], self.image.shape[0]))
            for cluster in self.clusters:
                cluster_str = "{},{},{}-".format(cluster.color[0], cluster.color[1], cluster.color[2])
                for position in cluster.positions:
                    cluster_str += "({},{}),".format(position[1], -position[0] + self.image.shape[0])
                save_file.write(cluster_str[:-1] + "\n")

class Cluster:
    def __init__(self, image_dimensions, color, position):
        self.color = color
        self.position = position
        self.colors = []
        self.positions = []

    def update_values(self):
        self.color = np.array([int(np.average(np.array(self.colors)[:, i])) for i in range(3)])
        self.position = np.array([int(np.average(np.array(self.positions)[:, i])) for i in range(2)])

image = Image.open("./Images/smaller_wolf.jpg")
# image = Image.open("./Images/wolf.jpg")
# image = Image.open("./Images/small_turtle.jpg")
image_array = np.array(image)[:, :, :3]
print("image array:", image_array.shape)

image_plot = plt.imshow(image_array)
plt.show()

manager = ClusterManager(100, image_array)
num_iterations = 4

plot = plt.subplot(2, (num_iterations + 1) // 2, 1)
plot.imshow(image_array)

for i in range(num_iterations):
    print("iteration number", i + 1)
    for cluster in manager.clusters:
        print(len(cluster.positions), len(cluster.colors))
    print("total:", sum([len(cluster.positions) for cluster in manager.clusters]), sum([len(cluster.colors) for cluster in manager.clusters]), len(manager.clusters))
    manager.iterate()

    plot = plt.subplot(2, (num_iterations + 1) // 2, i + 1)
    image_array = manager.apply_compression()
    plot.imshow(image_array / 255)
    plt.scatter([cluster.position[1] for cluster in manager.clusters], [-cluster.position[0] + image_array.shape[0] for cluster in manager.clusters], c='r')
    image = Image.fromarray(image_array, "RGB")
    print("image dims:", image.width, image.height)
    plt.imsave("./wolf_images/output" + str(i) + ".jpeg", image_array / 255)
    manager.save_clusters(i)

plt.show()