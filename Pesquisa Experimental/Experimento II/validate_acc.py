# Código utilizado para implementar a validação do classificador de dígitos 
# em imagens ruidosas (para diferentes níveis de ruído).

import idx2numpy
from random import randint
from mlp import *

# Quantas validações serão realizadas para cada nivel de ruído
N_EXECUTIONS = 10

# Entradas e saídas do conjunto MNIST (treino e validação)
TRAIN_IMAGE_FILENAME = '../mnist/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME = '../mnist/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME = '../mnist/t10k-images-idx3-ubyte'
TEST_LABEL_FILENAME = '../mnist/t10k-labels-idx1-ubyte'

def add_noise(image_set, std):
    # Implementação do Modelo de Ruído Estacionário Aditivo Gaussiano
    #
    # Parâmetros:
    # -- image_set (array de numpy array de floats): Conjunto de dados para ser degradado;
    # - std (float): Desvio padrão da distribuição gaussiana - nível de ruído.
    
    # Gera uma cópia dos dados
    noisy_image_set = np.copy(image_set)
    size = len(image_set[0])

    # Iterando o conjunto de dados para degradação
    for i in range(len(image_set)):
        # Gerando os valores aleatórios
        noise = np.random.normal(0.0, std, size)

        # Adicionando o ruído
        noisy_image_set[i] += noise

    # Limitação dos valores no intervalo [0,1]
    np.clip(noisy_image_set,0,1,noisy_image_set) 
    return noisy_image_set

def read_mnist(standardize=True):
    # Função auxiliar para ler e formatar os dados do conjunto MNIST
    #
    # Parâmetro:
    # -- standardize (bool): Se configurada, limita os valores do conjunto ao intervalo [0,1]
    #
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    # Reformatando - transformando as imagens em vetores.
    x_train = train_images.reshape(60000, 784)
    x_test = test_images.reshape(10000, 784)
    
    # Padronização dos dados
    if standardize:
        x_train = x_train/255.0
        x_test = x_test/255.0

    # Formatando as saídas para One-hot encoded 
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))
    
    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1
    
    return x_train, y_train, x_test, y_test

# Leitura dos dados
print("Reading dataset")
x_train, y_train, x_test, y_test = read_mnist()

# Importando as configurações do classificador de imagens 
# (aqui, um classificador de exemplo qualquer)
print("Importing classifier config")
classifier = MLP(2)
classifier.init_layer(0,784,25)
classifier.init_layer(1,25,10)
classifier.import_weights("classifier_config")

# Níveis de ruído
std_lvl = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# Para cada nível de ruído são efetuadas N validações da classificação nas imagens ruidosas
print("Initing main loop...")
for j in range(N_EXECUTIONS):
    print(f"##### Execution number: {j} #####")
    for i in range(len(std_lvl)):
        print(f"Noise std level: {std_lvl[i]}")
        
        # Adicionando o ruido ao conjunto de dados
        noisy_x_test = add_noise(x_test,std_lvl[i])

        # Cálculo da acurácia
        acc = 0
        for i in range(10000):
            if (classifier.predict(noisy_x_test[i]).argmax() == y_test[i].argmax()):
                acc += 1
        print(f"-- Classifier on Noisy data\nAcc: {acc/len(x_test)}")