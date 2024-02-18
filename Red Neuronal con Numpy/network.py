import random 
import numpy as np 

class Network(object): 

    def __init__(self, sizes):
        self.num_layers = len(sizes) 
        self.sizes = sizes  
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.v_w = [np.zeros(w.shape) for w in self.weights]  # Velocidad para los pesos
        self.v_b = [np.zeros(b.shape) for b in self.biases]  # Velocidad para los sesgos
        self.momentum_rate = 0.9  # Tasa de inercia (momentum rate)

    def feedforward(self, a): #Propagación hacia adelante
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: #Convierte en una lista los datos de prueba (test_data) si es que hay y obtiene la longitud de estos
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data) #Convierte en una lista los datos de entrenamiento si es que hay y oobtiene la longitud de estos
        n = len(training_data)
        for j in range(epochs): #Itera sobre el número de épocas especificadas
            random.shuffle(training_data) # Mezcla aleatoriamente los datos de entrenamiento en cada época.
            mini_batches = [ # Divide los datos de entrenamiento
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:    # Itera sobre cada mini_batch y actualiza los pesos y sesgos
                self.update_mini_batch(mini_batch, eta)
            if test_data: # Si hay datos de prueba, usa print para mostrar la precisión de la red en los datos de prueba
                          
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j)) # Si no hay datos de prueba, muestra que la época ha sido completada
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.v_w = [self.momentum_rate * v_w - (eta / len(mini_batch)) * nw
                    for v_w, nw in zip(self.v_w, nabla_w)]
        self.v_b = [self.momentum_rate * v_b - (eta / len(mini_batch)) * nb
                    for v_b, nb in zip(self.v_b, nabla_b)]
        self.weights = [w + v_w for w, v_w in zip(self.weights, self.v_w)]
        self.biases = [b + v_b for b, v_b in zip(self.biases, self.v_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.CrossEntropy_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers): 
            z = zs[-l]  
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def CrossEntropy_derivative(self, output_activations, y):
        return ((output_activations - y) / (output_activations * (1 - output_activations)))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

