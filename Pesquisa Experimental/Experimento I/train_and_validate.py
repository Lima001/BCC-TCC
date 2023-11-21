# Código utilizado para implementar a rotina de treino e validação do 
# modelo MLP para classificação de dígitos manuscritos do conjunto de
# dados MNIST.
#
# Baseado em Ekman (2021).

from mlp import *
import idx2numpy

# Entradas e saídas do conjunto MNIST (treino e validação)
TRAIN_IMAGE_FILENAME = '../mnist/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME = '../mnist/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME = '../mnist/t10k-images-idx3-ubyte'
TEST_LABEL_FILENAME = '../mnist/t10k-labels-idx1-ubyte'


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
x_train, y_train, x_test, y_test = read_mnist()

# Criando a MLP definida pelo Experimento I - baseada em Ekman (2021)
mlp = MLP(2)
mlp.init_layer(0,784,25)
mlp.init_layer(1,25,10)

# Se quiser utilizar parâmetros de exemplo para verificar a validação do modelo, use os comandos abaixo
# Obs. Estes não os mesmos parâmetros utilizados para obtenção dos resultados destacados na pesquisa.
mlp.import_weights("classifier_config")
mlp.validade(x_test,y_test)


# Se quiser treinar e exportar uma rede própria, utilize os comando abaixos
#mlp.train(x_train,y_train,x_test,y_test,print_mse=True)
#mlp.export_weights()