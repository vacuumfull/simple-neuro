# модули numpy и pathlib должны быть установлены в питон
import numpy as np
import os
import gzip
import pathlib
import matplotlib.pyplot as plt


class FashionMnist:

    MNIST_LABELS_PATH = 'data/t10k-labels-idx1-ubyte.gz'
    MNIST_IMAGES_PATH = 'data/t10k-images-idx3-ubyte.gz'
    COUNT_VALUES = 784
    LABEL_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    def __init__(self):
        self.images, self.labels = self.load_mnist()

        # Инициализация весов случайными числами
        self.weight_i_h = np.random.uniform(-0.5, 0.5, (20, self.COUNT_VALUES))
        self.weight_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
        #print('self.weight_i_h', type(self.weight_i_h), self.weight_i_h.shape)
        # Инициализация смещений нулями
        self.bias_i_h = np.zeros((20, 1))
        self.bias_h_o = np.zeros((10, 1))
        #print('self.bias_i_h', type(self.bias_i_h), self.weight_i_h.shape)
        #print('self.bias_h_o', type(self.bias_h_o), self.bias_h_o.shape)

        # learn_rate - шаг обучения, epochs - количество итераций, nr_correct - отслеживает количество правильных предсказаний
        self.learn_rate = 0.1
        self.nr_correct = 0
        self.epochs = 20
    
    def learn(self):
        for epoch in range(self.epochs):

        # Чтобы numpy мог посчитать скалярное произведение, необходимо подготовить одномерные матрицы img и l; 
        # использование zip ползволяет обрабатывать соответствующие элементы из нескольких массивов одновременно
            print("self.images[0].shape", self.images[0].shape)
            print("self.labels[0].shape", self.labels[0].shape)
            for img, l in zip(self.images, self.labels):
                img.shape += (1,)
                l.shape += (1,)

                # Прямое распространение ввод -> скрытый слой: вес смещения ввода суммируется с произведениями вводов и их весов 
                # (произведения между собой так же суммируются)
                h_pre = self.bias_i_h + self.weight_i_h @ img
                # Применение функции активации "сигмоида":  np.exp - функция экспоненты - возведение числа Эйлера в степень,
                #  указанную в скобках. При возведении экспоненты в отрицательные степени - результат стремится к нулю, 
                # а при возведении в положительные - к бесконечности
                h = self.sigmoid(h_pre)
                # Прямое распространение скрытый слой -> вывод: вес смещения скрытого слоя суммируется с произведениями
                #  значений узлов скрытого слоя и их весов (произведения между собой так же суммируются)
                o_pre = self.bias_h_o + self.weight_h_o @ h
                # Применение функции активации "сигмоида"
                o = self.sigmoid(o_pre)

                # Стоимость ошибок, она же Функция потерь (на примере среднеквадратической ошибки; в данном коде эта переменная 
                # никак не используется и оставлена просто для наглядности)

                # e = 1 / len(o) * np.sum((l - o) ** 2, axis=0)

                # Если ячейка с максимальным значением в слое вывода совпадает с ячейкой с максимальным значением в одномерной
                #  матрице 
                # l - labels - то счётчик правильных ответов увеличивается
                self.nr_correct += int(np.argmax(o) == np.argmax(l))

                # Обратное распространение вывод -> скрытый слой (производная функции потерь)
                delta_o = (2/len(o)) * (o - l)
                # К весу от скрытого слоя до вывода (на каждый нейрон соответственно) добавляется произведение его правильности на шаг 
                # обучения: у ошибочных результатов показатель отрицательный и, следовательно, он вычитается.
                # Транспонирование матриц - необходимо для их умножения. Операция умножения двух матриц выполнима только в том случае, 
                # если число столбцов в первом сомножителе равно числу строк во втором; в этом случае говорят, что матрицы согласованы.
                self.weight_h_o += -self.learn_rate * delta_o @ np.transpose(h) 
                self.bias_h_o += -self.learn_rate * delta_o

                # Обратное распространение скрытый слой -> ввод (производная композитной функции -
                #  производная функции активации умноженная на производную функции потерь))
                delta_h = np.transpose(self.weight_h_o) @ delta_o * (h * (1 - h))
                self.weight_i_h += -self.learn_rate * delta_h @ np.transpose(img)
                self.bias_i_h += -self.learn_rate * delta_h

            # Показать точность прогнозов для текущей итерации обучения и сбросить счётчик правильных прогнозов
            print(f"Уверенность: {round((self.nr_correct / self.images.shape[0]) * 100, 2)}%")
            self.nr_correct = 0


    def load_mnist(self):
        """Load MNIST data from `path`"""
        _path = pathlib.Path(__file__).parent.absolute()
        with gzip.open(f"{_path}/{self.MNIST_LABELS_PATH}", 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)
            labels = np.eye(10)[labels]

        with gzip.open(f"{_path}/{self.MNIST_IMAGES_PATH}", 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 784)
            images = images.astype("float32") / 255
        return images, labels

    @staticmethod
    def sigmoid(arr):
        return 1 / (1 + np.exp(-arr))

    def result(self):
        while True:
            index = int(input("Введите число (0 - 9999): "))
            img = self.images[index]
            plt.imshow(img.reshape(28, 28), cmap="Greys")
            
            # Прямое распространение ввод -> скрытый слой
            h_pre = self.bias_i_h + self.weight_i_h @ img.reshape(self.COUNT_VALUES, 1)
            # Активация сигмоида
            h = self.sigmoid(h_pre)
            # Прямое распространение скрытый слой -> вывод
            o_pre = self.bias_h_o + self.weight_h_o @ h
            # Активация сигмоида
            o = self.sigmoid(o_pre)
            print('o', type(o), o.shape)
            print(o)

            # argmax возвращает порядковый номер самого большого элемента в массиве
            plt.title(f"Нейросеть считает, что на картинке {self.LABEL_NAMES[o.argmax()]}")
            plt.show()

if __name__ == '__main__':
    fashion_mnist = FashionMnist()
    images, labels = fashion_mnist.load_mnist()
    print("images", type(images), images, images.shape)
    fashion_mnist.learn()
    fashion_mnist.result()