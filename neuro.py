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
        self.ideal = [0, 1, 1, 0]
        self.SPEED = 0.7
        self.MOMENT = 0.3
        self.weights = [0.45, 0.78, -0.12, 0.13, 1.5, -2.3, 
                        0.45, 0.78, -0.12, 0.13, 1.5, -2.3,
                        0.45, 0.78, -0.12, 0.13, 1.5, -2.3,
                        0.45, 0.78, -0.12, 0.13, 1.5, -2.3]
        self.delta_weights = [0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0]
        self.errors = [0,0,0,0]

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


    def update_weights(self, iteration, o_outs, h_outs):

        deltaO1 = self.delta_output(self.ideal[0], o_outs[0])
        deltaO2 = self.delta_output(self.ideal[1], o_outs[1])
        deltaO3 = self.delta_output(self.ideal[2], o_outs[2])
        deltaO4 = self.delta_output(self.ideal[3], o_outs[3])

        deltah1 = self.delta_hidden(h_outs[0], self.weights[16], deltaO1)
        deltah2 = self.delta_hidden(h_outs[1], self.weights[17], deltaO1)
        deltah3 = self.delta_hidden(h_outs[2], self.weights[18], deltaO2)
        deltah4 = self.delta_hidden(h_outs[3], self.weights[19], deltaO2)
        deltah5 = self.delta_hidden(h_outs[4], self.weights[20], deltaO3)
        deltah6 = self.delta_hidden(h_outs[5], self.weights[21], deltaO3)
        deltah7 = self.delta_hidden(h_outs[6], self.weights[22], deltaO4)
        deltah8 = self.delta_hidden(h_outs[7], self.weights[23], deltaO4)

        GRADw1 = self.inputs[0] * deltah1
        GRADw2 = self.inputs[1] * deltah1
        GRADw3 = self.inputs[0] * deltah2
        GRADw4 = self.inputs[1] * deltah2
        GRADw5 = self.inputs[0] * deltah3
        GRADw6 = self.inputs[1] * deltah3
        GRADw7 = self.inputs[0] * deltah4
        GRADw8 = self.inputs[1] * deltah4
        GRADw9 = self.inputs[0] * deltah5
        GRADw10 = self.inputs[1] * deltah5
        GRADw11 = self.inputs[0] * deltah6
        GRADw12 = self.inputs[1] * deltah6
        GRADw13 = self.inputs[0] * deltah7
        GRADw14 = self.inputs[1] * deltah7
        GRADw15 = self.inputs[0] * deltah8
        GRADw16 = self.inputs[1] * deltah8

        GRADw17 = h_outs[0] * deltaO1
        GRADw18 = h_outs[1] * deltaO1
        GRADw19 = h_outs[2] * deltaO2
        GRADw20 = h_outs[3] * deltaO2
        GRADw21 = h_outs[4] * deltaO3
        GRADw22 = h_outs[5] * deltaO3
        GRADw23 = h_outs[6] * deltaO4
        GRADw24 = h_outs[7] * deltaO4


        if iteration ==  1:
            self.delta_weights[0] = self.delta_weight(GRADw1)
            self.delta_weights[1] = self.delta_weight(GRADw2)
            self.delta_weights[2] = self.delta_weight(GRADw3)
            self.delta_weights[3] = self.delta_weight(GRADw4)
            self.delta_weights[4] = self.delta_weight(GRADw5)
            self.delta_weights[5] = self.delta_weight(GRADw6)
            self.delta_weights[6] = self.delta_weight(GRADw7)
            self.delta_weights[7] = self.delta_weight(GRADw8)
            self.delta_weights[8] = self.delta_weight(GRADw9)
            self.delta_weights[9] = self.delta_weight(GRADw10)
            self.delta_weights[10] = self.delta_weight(GRADw11)
            self.delta_weights[11] = self.delta_weight(GRADw12)
            self.delta_weights[12] = self.delta_weight(GRADw13)
            self.delta_weights[13] = self.delta_weight(GRADw14)
            self.delta_weights[14] = self.delta_weight(GRADw15)
            self.delta_weights[15] = self.delta_weight(GRADw16)
            self.delta_weights[16] = self.delta_weight(GRADw17)
            self.delta_weights[17] = self.delta_weight(GRADw18)
            self.delta_weights[18] = self.delta_weight(GRADw19)
            self.delta_weights[19] = self.delta_weight(GRADw20)
            self.delta_weights[20] = self.delta_weight(GRADw21)
            self.delta_weights[21] = self.delta_weight(GRADw22)
            self.delta_weights[22] = self.delta_weight(GRADw23)
            self.delta_weights[23] = self.delta_weight(GRADw24)
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
            
            self.delta_weights[6] = self.delta_weight(
                 GRADw7, self.delta_weights[6])
            self.delta_weights[7] = self.delta_weight(
                GRADw8, self.delta_weights[7])
            self.delta_weights[8] = self.delta_weight(
                GRADw9, self.delta_weights[8])
            self.delta_weights[9] = self.delta_weight(
                GRADw10, self.delta_weights[9])
            self.delta_weights[10] = self.delta_weight(GRADw11,
                                                      self.delta_weights[10])
            self.delta_weights[11] = self.delta_weight(GRADw12,
                                                      self.delta_weights[11])

            self.delta_weights[12] = self.delta_weight(
                 GRADw13, self.delta_weights[12])
            self.delta_weights[13] = self.delta_weight(
                GRADw14, self.delta_weights[13])
            self.delta_weights[14] = self.delta_weight(
                GRADw15, self.delta_weights[14])
            self.delta_weights[15] = self.delta_weight(
                GRADw16, self.delta_weights[15])
            self.delta_weights[16] = self.delta_weight(GRADw17,
                                                      self.delta_weights[16])
            self.delta_weights[17] = self.delta_weight(GRADw18,
                                                      self.delta_weights[17])

            self.delta_weights[18] = self.delta_weight(
                GRADw19, self.delta_weights[18])
            self.delta_weights[19] = self.delta_weight(
                GRADw20, self.delta_weights[19])
            self.delta_weights[20] = self.delta_weight(
                GRADw21, self.delta_weights[20])
            self.delta_weights[21] = self.delta_weight(
                GRADw22, self.delta_weights[21])
            self.delta_weights[22] = self.delta_weight(GRADw23,
                                                       self.delta_weights[22])
            self.delta_weights[23] = self.delta_weight(GRADw24,
                                                       self.delta_weights[23])


        self.weights[0] = self.weights[0] + self.delta_weights[0]
        self.weights[1] = self.weights[1] + self.delta_weights[1]
        self.weights[2] = self.weights[2] + self.delta_weights[2]
        self.weights[3] = self.weights[3] + self.delta_weights[3]
        self.weights[4] = self.weights[4] + self.delta_weights[4]
        self.weights[5] = self.weights[5] + self.delta_weights[5]
        self.weights[6] = self.weights[6] + self.delta_weights[6]
        self.weights[7] = self.weights[7] + self.delta_weights[7]
        self.weights[8] = self.weights[8] + self.delta_weights[8]
        self.weights[9] = self.weights[9] + self.delta_weights[9]
        self.weights[10] = self.weights[10] + self.delta_weights[10]
        self.weights[11] = self.weights[11] + self.delta_weights[11]
        self.weights[12] = self.weights[12] + self.delta_weights[12]
        self.weights[13] = self.weights[13] + self.delta_weights[13]
        self.weights[14] = self.weights[14] + self.delta_weights[14]
        self.weights[15] = self.weights[15] + self.delta_weights[15]
        self.weights[16] = self.weights[16] + self.delta_weights[16]
        self.weights[17] = self.weights[17] + self.delta_weights[17]
        self.weights[18] = self.weights[18] + self.delta_weights[18]
        self.weights[19] = self.weights[19] + self.delta_weights[19]
        self.weights[20] = self.weights[20] + self.delta_weights[20]
        self.weights[21] = self.weights[21] + self.delta_weights[21]
        self.weights[22] = self.weights[22] + self.delta_weights[22]
        self.weights[23] = self.weights[23] + self.delta_weights[23]


    def result(self):
        
        out = 0

        for i in range(5000):

            h1inp = self.weights[0] * self.inputs[0] + self.weights[1] * self.inputs[1]
            h1out = self.sigmoid(h1inp)

            h2inp = self.weights[2] * self.inputs[0] + self.weights[3] * self.inputs[1]
            h2out = self.sigmoid(h2inp)

            h3inp = self.weights[4] * self.inputs[0] + self.weights[5] * self.inputs[1]
            h3out = self.sigmoid(h3inp)

            h4inp = self.weights[6] * self.inputs[0] + self.weights[7] * self.inputs[1]
            h4out = self.sigmoid(h4inp)

            h5inp = self.weights[8] * self.inputs[0] + self.weights[9] * self.inputs[1]
            h5out = self.sigmoid(h5inp)

            h6inp = self.weights[10] * self.inputs[0] + self.weights[11] * self.inputs[1]
            h6out = self.sigmoid(h6inp)

            h7inp = self.weights[12] * self.inputs[0] + self.weights[13] * self.inputs[1]
            h7out = self.sigmoid(h7inp)

            h8inp = self.weights[14] * self.inputs[0] + self.weights[15] * self.inputs[1]
            h8out = self.sigmoid(h8inp)

            h_outs = [h1out, h2out, h3out, h4out, h5out, h6out, h7out, h8out]


            o1inp = self.weights[16]*h1out + self.weights[17]*h2out
            o1out = self.sigmoid(o1inp)

            o2inp = self.weights[18]*h1out + self.weights[19]*h2out
            o2out = self.sigmoid(o2inp)

            o3inp = self.weights[20]*h1out + self.weights[21]*h2out
            o3out = self.sigmoid(o3inp)

            o4inp = self.weights[22]*h1out + self.weights[23]*h2out
            o4out = self.sigmoid(o4inp)

            o_outs = [o1out, o2out, o3out, o4out]

            self.errors[0] = self.get_mse([{'expected': self.ideal[0], 'actual':  o1out}])
            self.errors[1] = self.get_mse(
                [{'expected': self.ideal[1], 'actual':  o2out}])
            self.errors[2] = self.get_mse(
                [{'expected': self.ideal[2], 'actual':  o3out}])

            self.errors[3] = self.get_mse(
                 [{'expected': self.ideal[3], 'actual':  o4out}])

            self.update_weights(i, o_outs, h_outs)


            if i == 4999:
                out = o_outs

        print('sec result ',  out)
        print('errors ', self.errors)


if __name__ == '__main__':
    neuro = XORNeuro()
    neuro.result()
