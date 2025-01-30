from sklearn.neural_network import MLPClassifier
import numpy as np

# Пример кода однослойного перцептрона на Python
# Функция активации (шаговая функция)
def step_function(x):
    return np.where(x >= 0, 1, 0)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.W = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        return step_function(np.dot(self.W, x))

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Вставка смещения (bias)
                prediction = self.predict(xi)
                self.W += self.learning_rate * (target - prediction) * xi

# Данные для обучения (И, ИЛИ)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Операция И (AND)

perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

# Тестирование
for xi in X:
    xi_with_bias = np.insert(xi, 0, 1)  # Вставка смещения (bias) для тестирования
    print(f"{xi} -> {perceptron.predict(xi_with_bias)}")

# Пример кода многослойного перцептрона на Python с использованием библиотеки scikit-learn
# Данные для обучения (И, ИЛИ)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Операция И (AND)

# Создание и обучение MLP-классификатора
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=400, learning_rate_init=0.01, solver='adam')
mlp.fit(X, y)

# Тестирование
for xi in X:
    print(f"{xi} -> {mlp.predict([xi])[0]}")

#Скопируйте в код однослойного персептрона и попытайтесь решить задачу XOR на 10 000, 20 000, 50 000 эпохах.
# Решение:

#  Данные для обучения(XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1]) # Операция XOR

# Функция для обучения и тестирования персептрона
def train_test(epochs):
    # Создание и обучение MLP-классификатора  с одним слоем (без скрытых нейронов)
    mlp = MLPClassifier(hidden_layer_sizes=(), activation='relu', max_iter=epochs, solver='adam')
    mlp.fit(X, y)
# Тестирование
    for xi in X:
        print(f"{xi} -> {mlp.predict([xi])[0]}")

# Тестирование на 10 000, 20 000, 50 000 эпохах:
train_test(10000)
train_test(20000)
train_test(50000)

