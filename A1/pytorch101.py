import torch


def ola():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente está configurado corretamente no Google Colab.
  """
  print('Olá do pytorch101.py!')


def criar_tensor_de_exemplo():
  """
  Retorna um Tensor do torch de shape (3, 2) preenchido com zeros, exceto para
  o elemento (0, 1) que será 10 e o elemento (1, 0) que será 100.

  Entrada: None

  Retorno:
  - Tensor de shape (3, 2) como descrito acima.
  """
  x = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  x = torch.tensor([[0,10],[100,0],[0,0]])
  #############################################################################
  #                            FIM DO SEU CODIGO                              #
  #############################################################################
  return x


def alterar_tensor(x, indices, valores):
  """
  Altera um tensor x do PyTorch de acordo com os parâmetros indices e valores.
  Especificamente, indices é uma lista [(i0, j0), (i1, j1), ... ] de indices
  inteiros e valores é uma lista [v0, v1, ...] de valores. Esta rotina deve 
  alterar x modificando:

  x[i0, j0] = v0
  x[i1, j1] = v1

  e assim por diante.

  Se o mesmo par de indices aparecer múltiplas vezes em indices, você deve 
  ajustar x para o último.

  Entrada:
  - x: Um Tensor de shape (H, W)
  - indices: Uma lista de N tuplas [(i0, j0), (i1, j1), ..., ]
  - valores: Uma lista de N valores [v0, v1, ...]

  Retorno:
  - O tensor de entrada x
  """
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  for i in range(len(indices)):
    x[indices[i][0],indices[i][1]] = valores[i]

  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return x


def contar_elementos_do_tensor(x):
  """
  Conta o número  de elementos escalares de um tensor x.

  Por exemplo, um tensor de shape (10,) tem 10 elementos. Um tensor de shape 
  (3, 4) tem 12 elementos; um tensor de shape (2, 3, 4) tem 24 elementos, etc.

  Você não pode usar as rotinas torch.numel ou x.numel. O tensor de entrada não
  deve ser modificado.

  Entrada:
  - x: Um tensor de um shape qualquer

  Retorno:
  - num_elementos: Um inteiro indicando o número de elementos escalares em x
  """
  num_elementos = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #   Você NÃO PODE usar as rotinas prontas torch.numel(x) ou x.numel().      #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  num_elementos = 1
  for i in range(len(x.shape)):
    num_elementos = num_elementos * x.shape[i]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return num_elementos


def criar_tensor_de_pi(M, N):
  """
  Retorna um Tensor de shape (M, N) inteiramente preenchido com o valor 3.14

  Entrada:
  - M, N: Inteiros positivos indicando o shape do Tensor a ser criado

  Retorno:
  - x: Um tensor de shape (M, N) preenchido com o valor 3.14
  """
  x = None
  #############################################################################
  #       TODO: Implemente esta rotina. Ela deve ocupar uma linha.            #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  x = torch.full((M,N), 3.14)
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return x


def multiplos_de_dez(inicio, fim):
  """
  Retorna um Tensor de dtype torch.float64 que contém todos os múltiplos de
  dez (em ordem crescente) no intervalo [inicio:fim]. Se não houver múltiplos
  de dez nesse intervalo você deve retornar um tensor vazio de shape (0,).

  Entrada:
  - inicio, fim: Inteiros com inicio <= fim indicando o intervalo a ser criado.

  Retorno:
  - x: Tensor de dtype float64 contendo múltiplos de dez entre inicio e fim.
  """
  assert inicio <= fim
  x = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  x = torch.arange(start=inicio,end=fim,step=5,dtype=torch.float64)
  x = x[x % 10 == 0]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return x


def pratica_de_indexacao_de_fatias(x):
  """
  Dado um tensor bidimensional x, extraia e retorne diversos subtensores para
  praticar a indexação de fatias. Cada tensor deve ser criado usando uma única
  operação de indexação de fatia.

  O tensor de entrada não deve ser modificado.

  Entrada:
  - x: Tensor de shape (M, N) -- M linhas, N colunas com M >= 3 e N >= 5.

  Retorno: Uma tupla de:
  - ultima_linha: Tensor de shape (N,) contendo a última linha de x. Deve ser
    um tensor unidimensional.
  - terceira_coluna: Tensor de shape (M, 1) contendo a terceira coluna de x.
    Deve ser um tensor bidimensional.
  - primeiras_duas_linhas_tres_colunas: Tensor de shape (2, 3) contendo os
    dados nas primeiras duas linhas e primeiras três colunas de x.
  - linhas_pares_colunas_impares: Tensor bidimensional contendo os elementos
    nas linhas pares e colunas ímpares de x.
  """
  assert x.shape[0] >= 3
  assert x.shape[1] >= 5
  ultima_linha = None
  terceira_coluna = None
  primeiras_duas_linhas_tres_colunas = None
  linhas_pares_colunas_impares = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  ultima_linha = x[x.shape[0]-1, :]
  terceira_coluna = x[:, 2:3]
  primeiras_duas_linhas_tres_colunas = x[:2,:3]
  linhas_pares_colunas_impares = x[::2, 1::2]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  saida = (
    ultima_linha,
    terceira_coluna,
    primeiras_duas_linhas_tres_colunas,
    linhas_pares_colunas_impares,
  )
  return saida


def pratica_atribuicao_de_fatias(x):
  """
  Dado um tensor bidimensional de shape (M, N) com M >= 4, N >= 6, altere suas
  4 primeiras linhas e 6 primeiras colunas de modo que elas sejam iguais a:

  [0 1 2 2 2 2]
  [0 1 2 2 2 2]
  [3 4 3 4 5 5]
  [3 4 3 4 5 5]


  Sua implementação deve obedecer ao seguinte:
  - Você deve alterar o próprio tensor x e retorná-lo
  - Você só deve modificar as primeiras 4 linhas e as primeiras 6 colunas; todos 
    os outros elementos devem permanecer inalterados
  - Você só pode alterar o tensor usando operações de atribuição de fatia, onde 
    você atribui um número inteiro a uma fatia do tensor
  - Você deve usar <= 6 operações de fatiamento para alcançar o resultado desejado
    
  Entrada:
  - x: Um tensor de shape (M, N) com M >= 4 e N >= 6

  Retorno: x
  """
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  y = torch.tensor([[0, 1, 2, 2, 2, 2],  [0, 1, 2, 2, 2, 2],  [3, 4, 3, 4, 5, 5],  [3, 4, 3, 4, 5, 5]])
  for i in range(4):
    x[i,0:6] = y[i,:]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return x


def embaralhar_colunas(x):
  """
  Reordena as colunas de um tensor de entrada conforme descrito abaixo.

  Sua implementação deve construir o tensor de saída usando uma única operação
  de indexação por arranjo de inteiros. O tensor de entrada não deve ser alterado.
    
  Entrada:
  - x: Um tensor de shape (M, N) com N >= 3

  Retorno: Um tensor y de shape (M, 4) onde:
  - As primeiras duas colunas de y são cópias da primeira coluna de x
  - A terceira coluna de y é igual a terceira coluna de x
  - A quarta coluna de y é igual a segunda coluna de x
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  xdesejadocol = torch.tensor([0, 0, 2, 1])
  y = x[:, xdesejadocol]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def inverter_linhas(x):
  """
  Inverte as linhas do tensor de entrada.

  Sua implementação deve construir o tensor de saída usando uma única operação
  de indexação por arranjo de inteiros. O tensor de entrada não deve ser alterado.

  Entrada:
  - x: Um tensor de shape (M, N)

  Retorno: Um tensor y de shape (M, N) que é igual a x mas com as linhas
           invertidas; ou seja, a primeira linha de y é igual a última linha
           de x, a segunda linha de y é igual a penúltima linha de x, etc.
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  xdesejadolin = torch.arange(x.shape[0])
  xdesejadolin = xdesejadolin.flip(0)
  y = x[xdesejadolin]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def pegar_um_elemento_por_coluna(x):
  """
  Construa um novo tensor escolhendo um elemento de cada coluna do
  tensor de entrada conforme descrito abaixo.    

  O tensor de entrada não deve ser alterado.

  Entrada:
  - x: Um tensor de shape (M, N) com M >= 4 e N >= 3.

  Retorno: Um tensor y de shape (3,) tal que:
  - O primeiro elemento de y é o segundo elemento da primeira coluna de x
  - O segundo elemento de y é o primeiro elemento da segunda coluna de x
  - O terceiro elemento de y é o quarto elemento da terceira coluna de x
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  xdesejadolin = [1, 0, 3]
  xdesejadocol = [0, 1, 2]
  y = x[xdesejadolin,xdesejadocol]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def contar_entradas_negativas(x):
  """
  Retorna o número de valores negativos no tensor de entrada x.

  Sua implementação deve efetuar somente uma única operação de indexação no 
  tensor de entrada. Você não deve usar nenhum loop explícito. O tensor de 
  entrada não deve ser alterado.

  Entrada:
  - x: Um tensor de qualquer shape

  Retorno:
  - num_neg: Inteiro indicando o número de valores negativos em x
  """
  num_neg = 0
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  mask = (x < 0)
  num_neg = x[mask].shape[0]
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return num_neg


def cria_one_hot(x):
  """
  Constrói um tensor de vetores one-hot a partir de um lista de inteiros do Python.

  Entrada:
  - x: Um lista de N inteiros

  Retorno:
  - y: Um tensor de shape (N, C) e onde C = 1 + max(x) é um a mais que o valor 
    máximo em x. A n-ésima linha de y é uma representação em vetor one-hot de 
    x[n]; Em outras palavras, se x[n] = c, então y[n, c] = 1; todos os outros 
    elementos de y são zeros. O dtype de y deve ser torch.float32.
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  desejado = torch.arange(len(x))
  col = max(x)+1
  lin = len(x)
  y = torch.zeros([lin,col],dtype= torch.float32)
  y[desejado,x] = 1
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def pratica_remodelagem(x):
  """
  Dado um tensor de entrada x de shape (24,), retorna um tensor remodelado y 
  de shape (3, 8) tal que

  y = [
    [x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
    [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
    [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]],
  ]

  Você deve contruir y realizando uma sequência de operações de remodelagem
  (view, t, transpose, permute, contiguous, reshape, etc). O tensor de
  entrada não deve ser alterado.

  Entrada:
  - x: Um tensor de shape (24,)

  Retorno:
  - y: Uma versão remodelada de x de shape (3, 8) conforme descrito acima.
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  x_test = x.view((2,3,4))
  y = x_test.transpose(0, 1).contiguous().view(3,8)
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def zero_linha_min(x):
  """
  Retorna uma cópia de x, onde o valor mínimo de cada linha foi alterado para 0.

  Por exemplo, se x é:
  x = torch.tensor([[
        [10, 20, 30],
        [ 2,  5,  1]
      ]])

  Então y = zero_linha_min(x) deve ser:
  torch.tensor([
    [0, 20, 30],
    [2,  5,  0]
  ])

  Sua implementação deve usar operações de redução e indexação; você não deve 
  usar nenhum loop explícito. O tensor de entrada não deve ser modificado.

  Entrada:
  - x: Tensor de shape (M, N)

  Retorno:
  - y: Tensor de shape (M, N) que é uma cópia de x, exceto que o valor mínimo
       de cada linha é substituído por 0.
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  y = torch.clone(x)
  x_vals_lin ,x_ind_col = x.min(dim=1)
  x_ind_lin = torch.arange(x.shape[0])
  y[x_ind_lin,x_ind_col] = 0
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def multiplicacao_de_matriz_em_lote(x, y, usa_loop=True):
  """
  Executa a multiplicação da matriz em lote entre o tensor x de shape (B, N, M) 
  e o tensor y de shape (B, M, P).    

  Se usa_loop = True, você deve usar um loop explícito sobre a dimensão de 
  lote B. Se usa_loop = False, então você deve calcular a multiplicação da 
  matriz em lote sem um loop explícito usando um único operador do PyTorch.

  Entrada:
  - x: Tensor de shape (B, N, M)
  - y: Tensor de shape (B, M, P)
  - usa_loop: Se deve usar um loop explícito do Python.

  Dica: torch.stack, bmm

  Retorno:
  - z: Tensor de shape (B, N, P) onde z[i] de shape (N, P) é o resultado da
       multiplicação de matriz entre x[i] de shape (N, M) e y[i] de shape
       (M, P). Deve ter o mesmo dtype que x.
  """
  z = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  T = x.shape[0]
  W = x.shape[1]
  D = y.shape[2]

  if(usa_loop == True):
    z = torch.empty(T, W, D)
    for i in range(T):
      z[i] = x[i].mm(y[i])
  else:
    z = torch.bmm(x,y)
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return z


def normaliza_colunas(x):
  """
  Normaliza as colunas da matriz x subtraindo a média e dividindo pelo desvio 
  padrão de cada coluna. Você deve retornar um novo tensor; o tensor de entrada 
  não deve ser modificado.

  Mais concretamente, dado um tensor de entrada x de shape (M, N), produza um 
  tensor de saída y de shape (M, N) onde y[i, j] = (x[i, j] - mu_j) / sigma_j, 
  onde mu_j é a média da coluna x[:, j].

  Sua implementação não deve usar nenhum loop explícito do Python (incluindo 
  compreensões de lista/conjunto/etc); você só pode usar operações aritméticas 
  básicas em tensores (+, -, *, /, **, sqrt), a função de redução de soma e 
  operações de remodelagem para facilitar a transmissão. Você não deve usar 
  torch.mean, torch.std ou suas variantes de método de instância x.mean, x.std.    

  Entrada:
  - x: Tensor de shape (M, N).

  Retorno:
  - y: Tensor de shape (M, N) conforme descrito acima. Deve ter o mesmo dtype 
    que a entrada x.
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  y = x.clone()
  med = torch.mean(y,dim=0)
  std = torch.std(y,dim=0)
  y = (y-med) / std
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y


def mm_na_cpu(x, w):
  """
  (função auxiliar) Executa multiplicação de matriz na CPU.
  POR FAVOR, NÃO EDITE ESTA CHAMADA DE FUNÇÃO.

  Entrada:
  - x: Tensor de shape (A, B), na CPU
  - w: Tensor de shape (B, C), na CPU

  Retorno:
  - y: Tensor de shape (A, C) conforme descrito acima. Não deve estar na GPU.
  """
  y = x.mm(w)
  return y


def mm_na_gpu(x, w):
  """
  Executa multiplicação de matriz na GPU.

  Especificamente, você deve (i) primeiro colocar cada entrada na GPU e, em seguida 
  (ii) executar a operação de multiplicação da matriz. Por fim, (iii) retorne o 
  resultado final, que está na CPU para uma substituição justa in-place com mm_na_cpu.

  Ao mover o tensor para GPU, POR FAVOR, use a operação "sua_instância_de_tensor.cuda()".

  Entrada:
  - x: Tensor de shape (A, B), na CPU
  - w: Tensor de shape (B, C), na CPU

  Retorno:
  - y: Tensor de shape (A, C) conforme descrito acima. Não deve estar na GPU.
  """
  y = None
  #############################################################################
  #                    TODO: Implemente esta rotina                           #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  temp = None
  x = x.cuda()
  w = w.cuda()
  temp = torch.matmul(x,w)
  y = temp.cpu()
  #############################################################################
  #                            FIM DO SEU CÓDIGO                              #
  #############################################################################
  return y
