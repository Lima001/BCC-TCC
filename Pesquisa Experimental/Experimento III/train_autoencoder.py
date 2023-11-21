# Código utilizado para implementar os removedores de ruído autoencoder (para
# uma determinada época) aplicado aos dados do conjunto MNIST

import idx2numpy
from random import randint
from mlp_autoencoder import *

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
    # Função auxiliar para ler e formatar os dados de treino do conjunto MNIST.
    # Observe que nesse caso são necessárias apenas as imagens de entradas, uma vez
    # não existe preocupação com a classificação dos dígitos, nem com validação
    #
    # Parâmetro:
    # -- standardize (bool): Se configurada, limita os valores do conjunto ao intervalo [0,1]
    
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)

    # Reformatando - transformando as imagens em vetores.
    x_train = train_images.reshape(60000, 784)
    
    # Padronização dos dados
    if standardize:
        x_train = x_train/255.0
    
    return x_train

# Leitura dos dados
print("Reading dataset")
x_train = read_mnist()

# Níveis de ruído
std_lvl = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# Iniciando loop de treinamento do modelo de remoção de ruído
print("Initing main loop...")
for i in range(len(std_lvl)):
    print(f"Noise std level: {std_lvl[i]}")
    
    # Adicionando ruido no conjunto de treino
    noisy_x_train = add_noise(x_train,std_lvl[i])

    print("-- Traning autoencoder")
    # Definição da rede neural autoencoder para remoção de ruído. 
    # Observe o espaço latente de 64 dimensões sendo definido pela quantidade
    # de saídas da primeira camada de neurônio.
    denoiser = MLP_A(2)
    denoiser.init_layer(0,784,64)
    denoiser.init_layer(1,64,784)
    
    # Treinando modelo - observe que as saída apresentadas são as imagens originais.
    # Dessa forma, o algoritmo pode usar o MSE aproximado como uma medidade de otimização
    # e não irá executar o aprendizado supervisionado.
    denoiser.train(noisy_x_train,x_train,epochs=EPOCHS)
    
    # Descomente o código abaixo caso queira exportar a configuração treinada
    #denoiser.export_weights(f"denoiser_config_{i+1}")
    print("-- Successfully trained autoencoder")