#Moran Alvarado Luis Andrés
#Fecha: 18 de febrero del 2024
#Actualización: Dejamos solo la libreri pickle y conservamoe el exit en su lugar


import mnist_loader
import network
import pickle #Importamos esta librería, tras ejecutar el codigo vimos que daba error por falta de esta libreria
from PIL import Image

training_data , test_data, _ = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 90, 10, 0.01, test_data=test_data) #Bajamos las epocas para que no tarde mucho en probar
with open('miRed.pkl','wb') as file1:
	pickle.dump(net,file1)
exit() #Conservamos el exit en su lugar inicial
file1=open('miRed.pkl','rb')
net2 = pickle.load(file1)



a=aplana(Imagen)
resultado = net.feedforward(a)
print(resultado)
