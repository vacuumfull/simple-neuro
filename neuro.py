import math


I1=1 
I2=0
O1ideal=1

w1=0.45 
w2=0.78 
w3=-0.12 
w4=0.13 
w5=1.5 
w6=-2.3

class XORNeuro:

    def __init__(self):
        self.inputs = [1, 0]
        self.ideal = 1
        self.SPEED = 0.7
        self.MOMENT = 0.3
        self.weights = [0.45, 0.78, -0.12, 0.13, 1.5, -2.3]
        self.delta_weights = [0, 0, 0, 0, 0, 0]
        self.err = 0

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative is True:
            return (1 - x)*x
        return 1/(1 + math.exp(-x))

    @staticmethod
    def get_mse(info: list):
        result = 0
        for i in info:  
            result += (i['expected'] - i['actual'])**2

        return result/len(info)


    def delta_weight(self, gradient, prev_delta=0):
        return self.SPEED*gradient + self.MOMENT*prev_delta

    def delta_output(self, expected, actual):
        return (expected - actual)*self.sigmoid(actual, True)


    def delta_hidden(self, out, weight, delta):
        return self.sigmoid(out, True)*(weight*delta)


    def update_weights(self, iteration, o1out, h1out, h2out):

        deltaO1 = self.delta_output(self.ideal, o1out)

        deltah1 = self.delta_hidden(h1out, self.weights[4], deltaO1)
        deltah2 = self.delta_hidden(h2out, self.weights[5], deltaO1)

        GRADw1 = self.inputs[0] * deltah1
        GRADw2 = self.inputs[0] * deltah2
        GRADw3 = self.inputs[1] * deltah1
        GRADw4 = self.inputs[1] * deltah2
        GRADw5 = h1out * deltaO1
        GRADw6 = h2out * deltaO1

        if iteration ==  1:
            self.delta_weights[0] = self.delta_weight(GRADw1)
            self.delta_weights[1] = self.delta_weight(GRADw2)
            self.delta_weights[2] = self.delta_weight(GRADw3)
            self.delta_weights[3] = self.delta_weight(GRADw4)
            self.delta_weights[4] = self.delta_weight(GRADw5)
            self.delta_weights[5] = self.delta_weight(GRADw6)
        else:
            self.delta_weights[0] = self.delta_weight(
                GRADw1, self.delta_weights[0])
            self.delta_weights[1] = self.delta_weight(
                GRADw2, self.delta_weights[1])
            self.delta_weights[2] = self.delta_weight(
                GRADw3, self.delta_weights[2])
            self.delta_weights[3] = self.delta_weight(
                GRADw4, self.delta_weights[3])
            self.delta_weights[4] = self.delta_weight(GRADw5, 
                                    self.delta_weights[4])
            self.delta_weights[5] = self.delta_weight(GRADw6, 
                                    self.delta_weights[5])

        self.weights[0] = self.weights[0] + self.delta_weights[0]
        self.weights[1] = self.weights[1] + self.delta_weights[1]
        self.weights[2] = self.weights[2] + self.delta_weights[2]
        self.weights[3] = self.weights[3] + self.delta_weights[3]
        self.weights[4] = self.weights[4] + self.delta_weights[4]
        self.weights[5] = self.weights[5] + self.delta_weights[5]


    def result(self):
        
        out = 0

        for i in range(5000):

            h1inp = self.weights[0] * self.inputs[0] + self.weights[2] * self.inputs[1]
            h1out = self.sigmoid(h1inp)

            h2inp = self.weights[1] * self.inputs[0] + self.weights[3] * self.inputs[1]
            h2out = self.sigmoid(h2inp)

            o1inp = self.weights[4]*h1out + self.weights[5]*h2out
            o1out = self.sigmoid(o1inp)

            self.err = self.get_mse([{'expected': self.ideal, 'actual':  o1out}])

            self.update_weights(i, o1out, h1out, h2out)
            
            if i == 4999:
                out = h1out

        print('sec result ',  out)
        print('error ', self.err)


if __name__ == '__main__':
    neuro = XORNeuro()
    neuro.result()
