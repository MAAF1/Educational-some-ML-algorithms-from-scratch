import numpy as np
from sigmoid_function import sigmoid

class NeuralNetwork:
    def __init__(self, hidden_weights, out_weights, activation) -> None: # our neural network basically has 2 layers 
        self.out_weights = out_weights                                   # but we can it's initialize the number of neurons according to the number of weights it gets 
        self.hidden_weights = hidden_weights 
        self.learning_rate = 0.5
        self.hidden_layer = NeuronLayer(self.hidden_weights,activation)
        self.out_layer = NeuronLayer(self.out_weights,activation)
        self.rows_hidden = hidden_weights.shape[0]
        self.cols_hidden = hidden_weights.shape[1]
        self.rows_out = out_weights.shape[0]
        self.cols_out = out_weights.shape[1]    
    
    def feed_forward(self,inputs,for_driv = 0): ## feed forward of the nerual network you can also get the hidden_layer output or the output_layer output
        hidden = self.hidden_layer.feed_forward(inputs)
        output = self.out_layer.feed_forward(hidden)
        if for_driv == 0:
            
            return output
        else:
            
            return hidden
    
    def compute_delta(self, inputs, target): ## compute derivative of each neuron by the previous one 
        predection = self.feed_forward(inputs)
        dE_dO_out = np.array(predection) - np.array(target)
        out_in = self.feed_forward(inputs, 1) 
        dE_dO_net = np.zeros((self.rows_out))
        
        for i in range(self.rows_out):
            dE_dO_net[i] = (dE_dO_out[i] * (self.out_layer.layer[i].derivative_neuron(out_in)))
        dE_dH_net = np.zeros((self.cols_out))
        
        for j in range(self.cols_out):
            c = 0
            for i in range(self.rows_out):   
                dO_net_dH_out = self.out_weights[i,j]  
                c += (dO_net_dH_out * dE_dO_net[i])          
            dE_dH_net[j] = self.hidden_layer.layer[j].derivative_neuron(inputs) * c
            dO_dW = np.zeros(self.out_weights.shape)
            for i in range(self.rows_out):
                for j in range(self.cols_out):
                    dO_dW[i,j] = dE_dO_net[i] *  out_in[j] 
            dH_dW = np.zeros(self.hidden_weights.shape)
            for i in range(self.rows_hidden):
                for j in range(self.cols_hidden):
                    dH_dW[i,j] = dE_dH_net[i] * inputs[j]
        return dO_dW , dH_dW
    
    def update_weights(self,out_gradient,hidden_gradient): # updating weights by the learning rate (all the weights of the neural network)
        self.hidden_weights  = self.hidden_weights - (self.learning_rate * hidden_gradient)
        self.out_weights = self.out_weights - (self.learning_rate * out_gradient)
        return self.out_weights , self.hidden_weights 
       

        

class NeuronLayer:
    def __init__(self,weights,activation) -> None:
        self.layer = [Neuron(0,0)] * weights.shape[0] # it's simply a list of neurons which initialized according to weights shape 
        self.activation = activation
        for i in range(len(self.layer)):
            self.layer[i] = (Neuron(weights[i,:],self.activation))     # careful the edges to the neuron is the weights of the neuron 
                                                                       # so basically you can calculate the weights using this approach backward
                                                                # like the first layer weights shape is 10 x 5 : so it means that their is 5 features as input
                                                                # and the hidden layer has 10 neurons.        
    def feed_forward(self,inputs):
        out_puts = []
        for i in range(len(self.layer)):
            out_puts.append(self.layer[i].cal_act())

        return out_puts

class Neuron:
    def __init__(self,weights,activation) -> None:
        self.weights = weights 
        self.activaion =activation
    
    def cal_net(self,inputs): ## adding all inputs which forwarded / putted into the neuron 
            out_net = 0
            for i in range(len(self.weights)):
                out_net += (inputs[i] * self.weights[i])
            
            return out_net
    
    def cal_act(self):
        
        out_final = self.cal_net()
        if self.activaion == "poly":   ## the activation function  
            return (out_final **2)
        elif self.activaion == "identity":
            return out_final
        elif self.activaion == "sigmoid":
            return sigmoid(out_final)
    
    def derivative_neuron(self): ## it's the derivative of the activation function of neuron itself
        
         
        doutact_dnet = self.cal_net()
        if self.activaion == "poly":
            return 2 *doutact_dnet
        elif self.activaion == "identity":
            return doutact_dnet
        elif self.activaion == "sigmoid":
            return (sigmoid(doutact_dnet) * (1 - sigmoid(doutact_dnet)))

def poly():
    hidden_layer_weights = np.array([[1,1],
                                     [2,1]])
    out_layer_weights = np.array([[2,1],
                                  [1,0]])
    nn = NeuralNetwork(hidden_layer_weights , out_layer_weights,"poly")
    print(nn.train_step([1,1],[290,14]))
def sigm():
    hidden_layer_weights = np.array([[0.1,0.1],
                                    [0.2,0.3],
                                    [0.1,0.3],
                                    [0.5,0.01]])
    out_layer_weights = np.array([[0.1,0.2,0.1,0.2],
                                  [0.1,0.1,0.1,0.5],
                                  [0.1,0.4,0.3,0.2]])          
    nn =NeuralNetwork(hidden_layer_weights , out_layer_weights,"sigmoid")

    print(nn.train_step(np.array([1,2]),np.array([0.4,0.7,0.6])))

if __name__ == "__main__":
    
    #poly()
    sigm()