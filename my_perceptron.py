import numpy as np

#input = w*x + b
def f(input):
	#boolean -> int
	return int(input > 0)

def run_perceptron(weights, data, labels, learning_rate=1):
    epoch_error = 0
    # Para cada instancia e label
    for x, y in zip(data, labels):
        # IMPLEMENTE AQUI A ATUALIZACAO DOS PESOS
        # x -> dados de uma linha da matriz
        # y -> saida correta (desejada)
        # o -> saida da rede (atual)
        o = f(np.sum(x * weights))
        weights = weights + learning_rate * (y - o) * x
        # abs -> modulo -> |x|
        epoch_error = epoch_error + abs(y - o)
        #print('Pesos: ',weights, 'Erros na Epoca:', epoch_error)

    #print('---------------------------------------------------------')    

    return weights, epoch_error

