import csv
import json
import numpy as np
from matplotlib import pyplot as plt
import random as rand
import pickle

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from keras.optimizers.legacy import Adam

import timeit

class Cluster:
    def __init__(self, color, points):
        self.points = points
        self.color = color

class Manager:
    def __init__(self):
        self.image_size = (0, 0)
        self.weights = []
        self.parameters = []
        self.clusters = []
        # for neural net format
        self.models = []
        self.dimensions = []

    def read_saved_data(self, path):
        with open(path) as save_file:
            csv_reader = csv.reader(save_file, delimiter="-")
            for i, row in enumerate(csv_reader):
                if i == 0:
                    row = row[0]
                    data = row.split(',')
                    self.image_size = (int(data[0]), int(data[1]))
                    continue
                color = [int(num) for num in row[0].split(",")]
                cluster = Cluster(color, [])
                for position in row[1][1:-1].split("),("):
                    cluster.points.append([int(num) / self.image_size[i] for i, num in enumerate(position.split(','))])
                self.clusters.append(cluster)

    def load_models(self, path):
        for i in range(len(self.clusters)):
            file_path = path + "/model" + str(i) + ".pkl"
            print("file path:", file_path)
            self.models.append(pickle.load(open(file_path, 'rb')))
            

    def recreate_image(self):
        image = np.zeros((self.image_size[0], self.image_size[1], 3))
        for cluster in self.clusters:
            for point in cluster.points:
                image[int(point[0] * self.image_size[0]) - 1][int(point[1] * self.image_size[1]) - 1] = cluster.color
        return image
    
    def load_dimensions(self, path):
        dimensions = []
        with open(path + '/dimensions', 'r') as dimension_file:
            dimensions_str = dimension_file.read()
            for saved_dimension in dimensions_str.split():
                dimension = []
                for i, num in enumerate(saved_dimension.split(',')):
                    dimension.append(int(float(num) * self.image_size[i>=2]))
                dimensions.append(dimension)
        self.dimensions = dimensions
            
    def create_train_and_label(self, included_clusters, excluded_clusters, include_weight, exclude_weight):
        inc_x, exc_x, y = [], [], []
        for cluster in included_clusters:
            inc_x += cluster.points[::include_weight]

        min_include_0 = min(inc_x, key=lambda x: x[0])[0] - 0.1
        max_include_0 = max(inc_x, key=lambda x: x[0])[0] + 0.1
        min_include_1 = min(inc_x, key=lambda x: x[1])[1] - 0.1
        max_include_1 = max(inc_x, key=lambda x: x[1])[1] + 0.1

        for cluster in excluded_clusters:
            exc_x += cluster.points[::exclude_weight]

        exc_x = list(filter(lambda point: point[0] > min_include_0 and point[0] < max_include_0 and point[1] > min_include_1 and point[1] < max_include_1, exc_x))

        # plt.subplot(2, 1, 1)
        # plt.scatter([point[0] for point in inc_x], [point[1] for point in inc_x], marker='o')
        # plt.scatter([point[0] for point in exc_x], [point[1] for point in exc_x], marker='x')

        x = inc_x + exc_x
        y += [1] * len(inc_x)
        y += [0] * len(exc_x)
        x = np.array(x)
        y = np.array(y)
        return x, y, [min_include_0, max_include_0, min_include_1, max_include_1]

    def feed_clusters(self):
        time = timeit.default_timer()
        for i in range(len(self.clusters)):
            # self.fit_equation_tf([self.clusters[i]], self.clusters[i+1:] + self.clusters[:i])
            self.fit_model([self.clusters[i]], self.clusters[i + 1:] + self.clusters[:i])
            curr_time = timeit.default_timer()
            print("completed equation {} at time {:2f}. Expected completion in {:2f}".format(i, (curr_time - time) / 60, ((curr_time - time) / (i + 1) * (len(self.clusters) - i)) / 60))
    
    def generate_dimensions(self, cluster):
        min_include_0 = min(cluster.points, key=lambda x: x[0])[0] - 0.1
        max_include_0 = max(cluster.points, key=lambda x: x[0])[0] + 0.1
        min_include_1 = min(cluster.points, key=lambda x: x[1])[1] - 0.1
        max_include_1 = max(cluster.points, key=lambda x: x[1])[1] + 0.1
        return [min_include_0, max_include_0, min_include_1, max_include_1]

    def fit_equation(self, included_clusters, excluded_clusters):
        plt.figure(figsize=(20, 20))
        x_train, y_train = self.create_train_and_label(included_clusters, excluded_clusters)
        poly = PolynomialFeatures(degree = 2, include_bias=True)
        x_poly = poly.fit_transform(x_train, y_train)
        model = LogisticRegression(max_iter=500, penalty=None, solver='sag')
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(x_poly, y_train, sample_weight=sample_weights)
        self.test_model(model, poly)
        self.weights.append(model.coef_)
        self.parameters.append(self.correct_scikit_names(poly.get_feature_names_out()))
        print("fit with score of", model.score(x_poly, y_train), "and parameters of", model.coef_)

    def fit_model(self, included_clusters, excluded_clusters):
        x_train, y_train, dimensions = self.create_train_and_label(included_clusters, excluded_clusters, 1, 2)
        test_dimensions = [dimensions[i] * self.image_size[i>=2] for i in range(4)]
        best_score = -1
        best_model = None
        times_converged = 0
        partial_converged = 0
        iterations = 0
        while times_converged < 1 and partial_converged < 5 and iterations < 15:
            # model = MLPClassifier((80, 3, 80), activation='relu', max_iter=15000, solver='adam', alpha=0, verbose=True, tol=0.0001, n_iter_no_change=500, learning_rate_init=0.002, warm_start=False, batch_size=2000)
            model = MLPClassifier((80, 3, 80), activation='relu', max_iter=40000, solver='adam', alpha=0, verbose=False, tol=0.0001, n_iter_no_change=5000, learning_rate_init=0.002, warm_start=False, batch_size=2000)
            model.fit(x_train, y_train)

            score = model.score(x_train, y_train)
            if score > 0.92:
                times_converged += 1
            if score > 0.85:
                partial_converged += 1
            print("iteration {} with score of {}".format(iterations, score))
            if score > best_score:
                best_model = model
                best_score = score
            iterations += 1
            # self.test_model(model, test_dimensions, None)

        print("best score is {}".format(best_model.score(x_train, y_train)))
        # self.test_model(best_model, dimensions, None)
        self.models.append(best_model)
        self.dimensions.append(dimensions)

    def test_model(self, model, dimensions, poly):
        print("testing model that has dimensions of", dimensions, [dimensions[0] / self.image_size[0], dimensions[1] / self.image_size[0], dimensions[2] / self.image_size[1], dimensions[3] / self.image_size[1]])
        t, f = [], []
        for i in range(max(0, int(dimensions[0])), int(dimensions[1])):
            for k in range(max(0, int(dimensions[2])), int(dimensions[3])):
                # point = poly.fit_transform(np.array([[i, k]]))
                point = np.array([[i / self.image_size[0], k / self.image_size[1]]])
                if model.predict(point):
                    t.append([i / self.image_size[0], k / self.image_size[1]])
                else:
                    f.append([i / self.image_size[0], k / self.image_size[1]])
        plt.subplot(2, 1, 2)
        plt.scatter([point[0] for point in t], [point[1] for point in t], color='b')
        plt.scatter([point[0] for point in f], [point[1] for point in f], color='r')
        plt.show()

    def draw_clusters(self):
        for cluster in self.clusters:
            plt.scatter([point[0] for point in cluster.points], [point[1] for point in cluster.points], color=(cluster.color[0]/255, cluster.color[1]/255, cluster.color[2]/255))
        plt.show()

    def test_models(self, folder_name):
        self.load_dimensions(folder_name)
        for model_index in range(0, len(self.clusters)):
            with open(folder_name + '/model{}.pkl'.format(model_index), 'rb') as f:
                model = pickle.load(f)
                inside_points = []
                for i in range(max(0, self.dimensions[model_index][0]), min(self.dimensions[model_index][1], self.image_size[0])):
                    for k in range(max(0, self.dimensions[model_index][2]), min(self.dimensions[model_index][3], self.image_size[1])):
                        if model.predict_proba(np.array([[i / self.image_size[0], k / self.image_size[1]]]))[0][1] > 0.3:
                            inside_points.append((i / self.image_size[0], k / self.image_size[1]))
            plt.scatter([point[0] for point in inside_points], [point[1] for point in inside_points], color=[self.clusters[model_index].color[i] / 255 for i in range(3)])
            # plt.scatter([point[0] for point in outside_points], [point[1] for point in outside_points], color=[0.9, 0.9, 0.9])
        plt.show()
        
    def test_model_tf(self, model):
        grid = np.array([[i, k] for k in range(self.image_size[1]) for i in range(self.image_size[0])])
        output = model.predict(grid)
        output = output.reshape(-1, self.image_size[1])
        positive = []
        negative = []
        for i in range(self.image_size[0]):
            for k in range(self.image_size[1]):
                if output[i][k] > 0.5:
                    positive.append([i, k])
                else:
                    negative.append([i, k])
        plt.subplot(2, 1, 2)
        plt.scatter([point[0] for point in positive], [point[1] for point in positive], color='b')
        plt.scatter([point[0] for point in negative], [point[1] for point in negative], color='r')
        plt.show()

    def convert_3layer_to_string(self, model, dimensions, index):
        x_vals = "c_{{{}}}=x*{}+{}".format(index, str(list(np.round(model.coefs_[0][0], decimals=4))), str(list(np.round(model.intercepts_[0], decimals=4))))
        y_vals = "v_{{{}}}=y*{}".format(index, str(list(np.round(model.coefs_[0][1], decimals=4))))
        inequality = "\\{" + "{}<x<{}".format(max(0,dimensions[0]), min(self.image_size[0],dimensions[1])) + "\\}\\{" + "{}<y<{}".format(max(0,dimensions[2]), min(self.image_size[1],dimensions[3])) + "\\}"
        total = "s(\\total({} * s((c_{{{}}} + v_{{{}}}))) + {}) > 0.5".format(str(list(np.round([arr[0] for arr in model.coefs_[1]], decimals=4))), index, index, str(np.round(model.intercepts_[1][0], decimals=4))) + inequality

        # print("equations:")
        # print("x_vals:", x_vals)
        # print("y_vals:", y_vals)
        # print("total:", total)
        return x_vals, y_vals, total
    
    # first layer: x, y
    # second layer: w[0]x + w[1]y + b
    # third layer: w[0]

    def convert_5layer_to_string(self, model, dimensions, index):
        # temporary probably:
        dimensions = np.round(self.generate_dimensions(self.clusters[index]), decimals=4)
        # test_dimensions = [dimensions[i] * self.image_size[i>=2] for i in range(4)]
        # self.test_model(model, test_dimensions, None)
        layer_1 = "n_{{{}}}=s(x*{}+y*{}+{})".format(index, str(list(np.round(model.coefs_[0][0], decimals=4))), str(list(np.round(model.coefs_[0][1], decimals=4))), str(list(np.round(model.intercepts_[0], decimals=4))))
        layer_2 = []
        for i in range(len(model.coefs_[1][0])):
            layer_2.append("{}_{{{}}}=s(\\total(n_{{{}}}*{})+{})".format(chr(i+65), index, index, str(list(np.round([arr[i] for arr in model.coefs_[1]], decimals=4))), str(np.round(model.intercepts_[1][i], decimals=4))))

        layer_3 = "q_{{{}}}=s({}".format(index, str(list(np.round(model.intercepts_[2], decimals=4))))
        for i, node in enumerate(layer_2):
            layer_3 += "+{}_{{{}}}*{}".format(node[0], index, str(list(np.round(model.coefs_[2][i], decimals=4))))
        layer_3 += ")"

        layer_4 = "s(\\total({}*q_{{{}}})+{})>0.5".format(str([np.round(coef[0], decimals=4) for coef in model.coefs_[3]]), index, str(list(np.round(model.intercepts_[3], decimals=4))))
        inequality = "\\{" + "{}<x<{}".format(max(0,dimensions[0]), min(1,dimensions[1])) + "\\}\\{" + "{}<y<{}".format(max(0,dimensions[2]), min(1,dimensions[3])) + "\\}"
        layer_4 += inequality
        return layer_1, layer_2, layer_3, layer_4
    
    
    def save_neural_net(self, index):
        print("attempting save")
        equations_data = {'equations' : []}
        for i, model in enumerate(self.models[:]):
            x_equal, y_equal, total_equal = self.convert_3layer_to_string(model, self.dimensions[i], i)
            equation = {'x_equal' : x_equal, 'y_equal' : y_equal, 'total_equal' : total_equal, 'color' : self.clusters[i].color}
            equations_data['equations'].append(equation)
            print("saved:", i)

            with open('wolf_models/model{}.pkl'.format(i),'wb') as file:
                pickle.dump(model, file)

            with open('wolf_models/dimensions', 'a') as file:
                file.write("{},{},{},{}\n".format(self.dimensions[i][0], self.dimensions[i][1], self.dimensions[i][2], self.dimensions[i][3]))

        with open("saved_wolf_equations{}.json".format(index), "w") as save_json:
            json.dump(equations_data, save_json)


    def save_deep_neural_net(self, index, folder_path, save_dims=True, save_pkl=True):
        equations_data = {'equations' : []}
        for i, model in enumerate(self.models[:]):
            layer1, layer2_list, layer3, layer4 = self.convert_5layer_to_string(model, self.dimensions[i], i)
            equation = {'layer1' : layer1, 'layer2list' : layer2_list, 'layer3' : layer3, 'layer4' : layer4, 'color' : self.clusters[i].color}
            equations_data['equations'].append(equation)
            print("saved:", i)

            # with open(folder_path + '/model{}.pkl'.format(i),'wb') as file:
            #     pickle.dump(model, file)

            # with open(folder_path + '/dimensions', 'a') as file:
            #     file.write("{},{},{},{}\n".format(self.dimensions[i][0], self.dimensions[i][1], self.dimensions[i][2], self.dimensions[i][3]))

        with open("saved_wolf_equations{}.json".format(index), "w") as save_json:
            json.dump(equations_data, save_json)

    def correct_scikit_names(self, parameter_names):
        parameter_names = '!'.join(parameter_names)
        parameter_names = parameter_names.replace('x0', 'x')
        parameter_names = parameter_names.replace('x1', 'y')
        parameter_names = parameter_names.replace(' ', '*')
        parameter_names = parameter_names.split('!')
        return parameter_names

    def convert_to_equation_str(self, parameter_list, weight_list):
        equation = str(weight_list[0])[:7] # add first constant value
        for parameter, weight in zip(parameter_list[1:], weight_list[1:]):
            equation += "+" + str(weight)[:7] + str(parameter)
        return equation + " > 0.5"

    def save_equations(self, index):
        equations_data = {'equations' : []}
        equations_data['image_size'] = self.image_size
        for i in range(len(self.parameters)):
            equation = self.convert_to_equation_str(self.parameters[i], self.weights[i][0])
            color = self.clusters[i].color
            equations_data['equations'].append({'equation' : equation, 'color' : color})

        with open("saved_equations{}.json".format(index), "w") as save_json:
            json.dump(equations_data, save_json)
        

    
manager = Manager()
manager.read_saved_data("./wolf_images/label3")
# manager.read_saved_data('mid_data/label9')
image = manager.recreate_image() / 255
plt.imshow(image)
plt.show()

manager.feed_clusters()
manager.save_neural_net(2)

# manager.load_dimensions("wolf_models")
# manager.load_models("wolf_models")
# manager.save_deep_neural_net(1, "wolf_models", False, False)

# manager.test_models('wolf_models')