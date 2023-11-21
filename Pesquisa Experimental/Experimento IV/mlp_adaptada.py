# Implementação do modelo Multilayer Perceptron (MLP) adaptado para remoção de ruído de dígitos MNIST
#
# Recapitulando a ideia do experimento: Adicionar a medida de erro da classificação no algoritmo de treino
# do modelo de remoção de ruído.
#
# O que muda para a implementação apresentada no Experimento III:
# -- Adição de parâmetros novos no método "train()" 
# -- -- omega e phi: São as configurações para a adaptação
# -- -- cl: É a rede de classificação de imagens (objeto Python)

import numpy as np
import os
import csv

# Definição de valores padrões para algumas configurações usadas pelos modelos. Quando não informados
# durante invocação do método para treinar a rede neural, esses valores são usados.
LEARNING_RATE = 0.01                                                        # Taxa de aprendizado (alpha)
EPOCHS = 10                                                                 # Épocas de treino

# Limites para o intervalo de geração de pesos. Por padrão, ao inicializar a rede neural os
# pesos são gerados aleatoriamentes a partir do intervalo [LOWER_BOUND, UPPER_BOUND]. Como
# observação, destaca-se que os biases são inicializados com valor 0. 
LOWER_BOUND = -0.1                                                          # Limite inferior       
UPPER_BOUND = 0.1                                                           # Limite superior

class A_MLP:

    def __init__(self, n_layers):
        self.n_layers = n_layers                                            # Quantidade de camadas (de neurônios)
        self.weights = [None for i in range(n_layers)]                      # Array para armazenar os pesos de cada camada
        self.outputs = [None for i in range(n_layers)]                      # Array auxilair - Armazena as saídas de cada camada - usado durante o algoritmo backpropagation
        self.errors = [None for i in range(n_layers)]                       # Array auxliar - Armazena  os termos de erro de cada neurônio - usado durante o algoritmo backpropagation

    def init_layer(self, ith, input_count, neuron_count, lb=LOWER_BOUND, ub=UPPER_BOUND):
        # Método para inicializar/configurar uma camada de neurônios da rede. 
        #
        # Parâmetros:
        # -- ith (int): Índice da i-ésima camada a ser inicializada, sendo 0 a primeira camada;
        # -- input_count (int): Quantidade de sinais de entrada dos neurônios da i-ésima camada;
        # -- neuron_count (int): Quantidade de neurônios da i-ésima camada;
        # -- lb (float): Limite inferior para geração dos pesos aleatórios;
        # -- ub (float): Limite superior para geração dos pesos aleatórios.

        # Inicializa a matriz de pesos da i-ésima camada. Por padrão, é uma matriz nula.
        # Obs. A matriz já considera a inclusão do bias.
        w = np.zeros((neuron_count,input_count+1))

        # Gerando os pesos aleatórios
        for i in range(neuron_count):
            for j in range(1, (input_count+1)):
                w[i][j] = np.random.uniform(lb, ub)

        # Configurando os arrays da rede neural
        self.weights[ith] = w

        # Arrays auxiliares são inicializados como vetores nulos (são usados durante o treinamento da rede)
        self.outputs[ith] = np.zeros(neuron_count)
        self.errors[ith] = np.zeros(neuron_count)

    def export_weights(self, directory="mlp_config"):
        # Método para exporar os parâmetros de um modelo para arquivos .csv.
        # Cada matriz de peso é exportada individualmente como um arquivo. Sendo assim, para cada
        # camada da rede são gerados um arquivo de configuração contendo os parâmetros do modelo.
        #
        # Parâmetros:
        # -- directory (str): Caminho do diretório em que serão salvos os parâmetros.

        # Verifica se o diretório informado já existe. Se não, ele é criado
        if (not os.path.isdir(directory)):
            os.mkdir(directory)
        
        # Exporta os parâmetros para seus respectivos arquivos.
        # Padrão de nomeação: weights_layer_{i}.csv
        # onde 'i' refere-se ao índice da respectiva camada que os parâmetros fazem parte
        for i in range(self.n_layers):
            with open(f"{directory}/weights_layer_{i}.csv", mode="w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.weights[i])

    def import_weights(self, directory="mlp_config"):
        # Método para importar os parâmetros de um modelo a partir de arquivos .csv.
        # A formatação dos arquivos deve seguir exatamente o padrão adotado pelo método
        # 'export_weights' dessa classe.
        #
        # Parâmetros:
        # -- directory (str): Caminho do diretório em que serão lidos os arquivos com os parâmetros.
        
        # Tratamento para inexistência do diretório informado
        if (not os.path.isdir(directory)):
            raise Exception("Configuration directory not found!")

        # Leitura e configuração dos parâmetros do modelo
        for i in range(self.n_layers):
            with open(f"{directory}/weights_layer_{i}.csv", mode="r") as f:
                reader = csv.reader(f)
                self.weights[i] = [np.array(row, dtype=float).tolist() for row in reader]

    def forward(self, x):
        # Implementa o forward pass do algoritmo backpropagation.
        # 
        # Parâmetros:
        # -- x (numpy array de floats): Um vetor de acordo com o formato de entrada aceito pela rede

        # Calculando as saídas para a camada de entrada
        for i, w in enumerate(self.weights[0]):
            z = np.dot(w,x)                                             # Vetor com os potências de ativação
            self.outputs[0][i] = np.tanh(z)                             # Aplicação da função de ativação

        # Calculando saídas para as camadas ocultas
        for j in range(1,self.n_layers-1):
            # Criando entrada para camada oculta - saídas da camada anterior mais x0=1.0 fixo (para cálculo do bias)
            hidden_output_array = np.concatenate((np.array([1.0]), self.outputs[j-1]))

            for i, w in enumerate(self.weights[j]):
                z = np.dot(w,hidden_output_array)                       # Vetor com os potências de ativação
                self.outputs[j][i] = np.tanh(z)                         # Aplicação da função de ativação
        
        # Calculando as saídas para a camada de saída
        
        # Criando entrada para camada oculta - saídas da camada anterior mais x0=1.0 fixo (para cálculo do bias)
        hidden_output_array = np.concatenate((np.array([1.0]), self.outputs[-2]))
        for i, w in enumerate(self.weights[-1]):
            z = np.dot(w,hidden_output_array)                           # Vetor com os potências de ativação
            self.outputs[-1][i] = 1.0 / (1.0 + np.exp(-z))              # Aplicação da função de ativação
    
    def backward(self, y_truth, c_error, phi, omega, print_mse=False):
        # Implementa o backward pass do algoritmo backpropagation.
        #
        # Obs. Utiliza como base os valores armazenados nos arrays auxiliares. Pressupõe que o método seja  
        # chamado após sua respectiva entrada ter sido processada pelo método 'foward()' dessa classe.
        #
        # Parâmetros:
        # -- y_truth (numpy array de floats): Saída esperada;
        # -- c_error (float): Erro da rede neural classificadora de imagens para a saída processada
        #                     modelo em questão;
        # -- phi, omega (floats): Configurações propostas para adaptação do modelo;
        # -- print_mse (bool): flag identificado se o erro MSE aproximado deve ser exibido .
        
        # Cálculando o termo de erro para os neurônios da camada de saída
        error_sum = 0.0                                                             # Erro MSE acumulado por neurônio
        for i, y in enumerate(self.outputs[-1]):
            error_sum += (y_truth[i] - y)**2                                        # Somando o erro MSE aproximado para um único neurônio
            
            # Cálculo de termo de erro (propriamente)
            error_prime = -(y_truth[i] - y)*phi - c_error*omega                       # ADAPTAÇÃO PROPOSTA - PONDERAÇÃO DO ERRO
            derivative = y * (1.0 - y)                                              # Derivada da função logsig
            self.errors[-1][i] = error_prime * derivative

        # Se a flag for configurada, imprime o erro MSE aproximado da rede
        if print_mse:
            print(f"MSE: {error_sum/len(y_truth)}")

        # Cálculando o termo de erro para os neurônios das camadas restantens
        for j in range(self.n_layers-2, -1, -1):
            for i, y in enumerate(self.outputs[j]):
                error_weights = []
                
                # Recuperando os termos de erros da camada posterior a atual
                for w in self.weights[j+1]:
                    error_weights.append(w[i+1])
                
                # Formatando para array do numpy 
                error_weight_array = np.array(error_weights)

                # Cálculo de termo de erro (propriamente)
                derivative = 1.0 - y**2                                                         # Derivada da função tanh
                weighted_error = np.dot(error_weight_array, self.errors[j+1])
                self.errors[j][i] = weighted_error * derivative

    def adjust_weights(self, x, lr=LEARNING_RATE):
        # Implementa o ajuste dos parâmetros do modelo realizado pelo algoritmo backpropagation.
        #
        # Obs. Utiliza como base os valores armazenados nos arrays auxiliares. Pressupõe que o método seja  
        # chamado após sua respectiva entrada ter sido processada pelo método 'backward()' dessa classe.
        #
        # Parâmetros:
        # -- x (numpy array de floats): Entrada da rede (necessário para atualizar os parâmetros da camada de entrada);
        # -- lr (float): taxa de aprendizado; 
        
        # Atualizando os parâmetros da camada de entrada
        for i, e in enumerate(self.errors[0]):
            self.weights[0][i] -= (x * lr * e)
    
        # Atualizando os parâmetros da camadas subsequentes
        hidden_output_array = None
        for j in range(1,self.n_layers):
            hidden_output_array = np.concatenate((np.array([1.0]), self.outputs[j-1]))

            for i, e in enumerate(self.errors[j]):
                self.weights[j][i] -= (hidden_output_array * lr * e)

    def train(self, x_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE, print_mse=False, phi=1.0, omega=0.0, cl =None):
        # Método para treinar a rede neural usando o algoritmo backpropagation estocástico.
        #
        # Obs. Esse método abstrai os métodos 'foward(), backward()' e 'adjust_weights()'
        #
        # Parâmetros:
        # -- x_train (array de numpy arrays): Entradas do conjunto de dados de treino;
        # -- y_train (array de numpy arrays): Saídas esperadas do conjunto de dados de treino;
        # -- epochs (int): Épocas de treino;
        # -- lr (float): Taxa de aprendizado;
        # -- print_mse (bool): Flag. Se configurada, é repassada ao método 'backward()' para que o erro aproximado seja 
        #                      exibido para cada iteração no conjunto de dados;
        # -- phi (float): Configuração proposta pelo experimento IV;
        # -- omega (float): Configuração proposta pelo experimento IV;
        # -- cl (Python Object - MLP Classificadora de Imagens);

        
        # Índices com a sequência de treino (inicialmente ordenado)
        index_train = list(range(len(x_train)))
        
        # Executa o processo de treino para cada época
        for i in range(epochs):
            # Trainamento

            # Define uma ordem aleatória para treinar usando os dados de treino
            np.random.shuffle(index_train)
            
            # Iterando sobre os dados de treino para realizar o processo de aprendizagem
            for j in index_train:
                # Adiciona x0=1.0 para comportar o bias
                x = np.concatenate((np.array([1.0]), x_train[j]))
                
                # Gerando a saída contendo a imagem (representação vetorial) pré-processada
                self.forward(x)
                
                #Calculando o erro do modelo de classificação para a saída processada
                # pelo modelo de remoção de ruído
                c_error = cl[0].get_error(self.outputs[-1], cl[1][j])

                # Ajuste do modelo
                self.backward(y_train[j], c_error, phi, omega, print_mse)
                self.adjust_weights(x, lr)

    
    def predict(self, x, verbose=False):
        # Método para realizar a inferência do modelo.
        #
        # Parâmetros:
        # -- x (numpy array de floats): vetor de entrada única (sem x0=1);
        # -- verbose (bool): Flag. Exibir informações de debug (entrada e saída).
        
        # Retorna:
        # -- vet0r corerspondente as saídas dos neurônios da última camada da rede.        
        
        xi = np.concatenate((np.array([1.0]), x))
        
        self.forward(xi)
            
        if verbose:
            print(f"Input: {x}")
            print(f"Output: {self.outputs[-1]}")
        
        return self.outputs[-1]