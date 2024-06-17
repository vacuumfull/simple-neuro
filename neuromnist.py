# модули numpy и pathlib должны быть установлены в питон
import numpy as np
import pathlib
import matplotlib.pyplot as plt


class NeuroMnist:

    MNIST_PATH = 'data/mnist.npz'

    def __init__(self):
        self.images, self.labels = self.get_mnist()

        # Инициализация весов случайными числами
        self.weight_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
        self.weight_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
        #print('self.weight_i_h', type(self.weight_i_h), self.weight_i_h.shape)
        # Инициализация смещений нулями
        self.bias_i_h = np.zeros((20, 1))
        self.bias_h_o = np.zeros((10, 1))
        #print('self.bias_i_h', type(self.bias_i_h), self.weight_i_h.shape)
        #print('self.bias_h_o', type(self.bias_h_o), self.bias_h_o.shape)

        # learn_rate - шаг обучения, epochs - количество итераций, nr_correct - отслеживает количество правильных предсказаний
        self.learn_rate = 0.2
        self.nr_correct = 0
        self.epochs = 3  


    # Объявление функции "get_mnist", которая будет возвращать значения, указанные ниже в команде return (images и labels).
    def get_mnist(self):

    # Из файла извлекается два массива: images (из ключа “x_train”) и labels (из ключа “y_train”).
    # x_train содержит изображения цифр, а y_train - соответствующие им метки (цифры от 0 до 9).
    # Примечание: предполагается, что файл mnist.npz размещён в папке data, которая находится в папке со скриптом.
        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/{self.MNIST_PATH}") as f:
            images, labels = f["x_train"], f["y_train"]

    # Преобразуем тип данных массива images в float32 и сожмём значения в диапазон от 0 до 1 путем деления на 255. 
            images = images.astype("float32") / 255

        # images - трёхмерный массив двухмерных картинок, [0] измерение это количество картинок, а измерения [1] и [2] 
        # - размерности по высоте и ширине. Умножив размерности [1] и [2] друг на друга - получается общее количество
        #  пикселей в изображении. На выходе получается двухмерный массив - матрица.
            images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2])) 

        # Здесь метки преобразуются в формат "one-hot encoding". Мы создаем матрицу размером 10x10, 
        # где каждая строка представляет одну метку (цифру от 0 до 9). Значение 1 в строке соответствует метке,
        #  а остальные значения равны 0. 
            labels = np.eye(10)[labels]

        # Функция возвращает два массива: images (обработанные изображения) и labels (one-hot encoded метки).
            return images, labels

    @staticmethod
    def sigmoid(arr):
        return 1 / (1 + np.exp(-arr))
        
    def learn(self):
        for epoch in range(self.epochs):

        # Чтобы numpy мог посчитать скалярное произведение, необходимо подготовить одномерные матрицы img и l; 
        # использование zip ползволяет обрабатывать соответствующие элементы из нескольких массивов одновременно
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


    def result(self):
        while True:
            index = int(input("Введите число (0 - 59999): "))
            img = self.images[index]
            plt.imshow(img.reshape(28, 28), cmap="Greys")
            
            # Прямое распространение ввод -> скрытый слой
            h_pre = self.bias_i_h + self.weight_i_h @ img.reshape(784, 1)
            # Активация сигмоида
            h = self.sigmoid(h_pre)
            # Прямое распространение скрытый слой -> вывод
            o_pre = self.bias_h_o + self.weight_h_o @ h
            # Активация сигмоида
            o = self.sigmoid(o_pre)
            print('o', type(o), o.shape)

            # argmax возвращает порядковый номер самого большого элемента в массиве
            plt.title(f"Нейросеть считает, что на картинке цифра {o.argmax()}")
            plt.show()


if __name__ == '__main__':
    neuro_mnist = NeuroMnist()
    neuro_mnist.learn()
    neuro_mnist.result()
