"""
Implementa uma Rede Neural de duas camadas no PyTorch.
AVISO: você NÃO DEVE usar ".to()" ou ".cuda()" em cada bloco de implementação.
"""
import torch
import random
import statistics


def ola_rede_duas_camadas():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente esteja configurado corretamente no Google Colab.    
  """
  print('Olá do rede_duas_camadas.py!')


# Módulos de classe de modelo que usaremos mais tarde: Não edite / modifique esta classe
class RedeDuasCamadas(object):
  def __init__(self, tamanho_entrada, tamanho_oculta, tamanho_saida,
               dtype=torch.float32, device='cuda', std=1e-4):
    """
    Inicializa o modelo. Pesos são inicializados com pequenos valores aleatórios 
    e vieses são inicializados com zero. Pesos e vieses são armazenados na variável 
    self.params, que é um dicionário com as seguintes chaves:               

    W1: Pesos da primeira camada; tem shape (D, H)
    b1: Vieses da primeira camada; tem shape (H,)
    W2: Pesos da segunda camada; tem shape (H, C)
    b2: Vieses da segunda camada; tem shape (C,)

    Entrada:
    - tamanho_entrada: A dimensão D dos dados de entrada.
    - tamanho_oculta: O número de neurônios H na camada oculta.
    - tamanho_saida: O número de categorias C.
    - dtype: Opcional, tipo de dados de cada parâmetro de peso.
    - device: Opcional, se os parâmetros de peso estão na GPU ou CPU.
    - std: Opcional, ajuste de escala dos parâmetros de peso.
    """
    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)

    self.params = {}
    self.params['W1'] = std * torch.randn(tamanho_entrada, tamanho_oculta, dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(tamanho_oculta, dtype=dtype, device=device)
    self.params['W2'] = std * torch.randn(tamanho_oculta, tamanho_saida, dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(tamanho_saida, dtype=dtype, device=device)

  def perda(self, X, y=None, reg=0.0):
    return rn_frente_tras(self.params, X, y, reg)

  def treinar(self, X, y, X_val, y_val,
            taxa_de_aprendizado=1e-3, decaimento_taxa_de_aprendizado=0.95,
            reg=5e-6, num_iteracoes=100,
            tamanho_lote=200, verbose=False):
    return rn_treinar(
            self.params,
            rn_frente_tras,
            rn_inferir,
            X, y, X_val, y_val,
            taxa_de_aprendizado, decaimento_taxa_de_aprendizado,
            reg, num_iteracoes, tamanho_lote, verbose)

  def inferir(self, X):
    return rn_inferir(self.params, rn_frente_tras, X)

  def salvar(self, path):
    torch.save(self.params, path)
    print("Salvo em {}".format(path))

  def carregar(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint
    print("carrega arquivo de checkpoint: {}".format(path))


def amostrar_lote(X, y, num_entradas, tamanho_lote):
  """
  Amostra de tamanho_lote elementos dos dados de treinamento e seus 
  rótulos correspondentes para usar nesta rodada de descida de gradiente.    
  """
  X_lote = None
  y_lote = None
  #########################################################################
  # TODO: Armazene os dados em X_lote e seus rótulos correspondentes em   #
  # y_lote; após a amostragem, X_lote deve ter shape (tamanho_lote, dim)  # 
  # e y_lote deve ter shape (tamanho_lote,)                               #
  #                                                                       #
  # Dica: Use torch.randint para gerar índices.                           #
  #########################################################################
  # Substitua a comando "pass" pelo seu código
  index = torch.randint(num_entradas, size = (tamanho_lote,))
  X_lote = X[index]
  y_lote = y[index]
  #########################################################################
  #                          FIM DO SEU CODIGO                            #
  #########################################################################
  
  return X_lote, y_lote


def rn_passo_para_frente(params, X):
    """
    O primeiro estágio de nossa implementação da rede neural: Executar o 
    passo para frente da rede para calcular as saídas da camada oculta e 
    pontuações de classificação. A arquitetura da rede deve ser:

    camada TC -> ReLU (rep_oculta) -> camada TC (pontuações)

    Como prática, NÃO permitiremos o uso das operações torch.relu e torch.nn 
    apenas neste momento (você pode usá-las na próxima tarefa).

    Entrada:
    - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
       um modelo. Deve ter as seguintes chaves com shape:
           W1: Pesos da primeira camada; tem shape (D, H)
           b1: Vieses da primeira camada; tem shape (H,)
           W2: Pesos da segunda camada; tem shape (H, C)
           b2: Vieses da segunda camada; tem shape (C,)
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: Uma tupla de:
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classificação para X
    - rep_oculta: Tensor de shape (N, H) contendo a representação da camada oculta
      para cada valor de entrada (depois da ReLU).
    """
    # Descompacte as variáveis do dicionário de parâmetros
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # Calcule o passo para frente
    pontuacoes = None
    rep_oculta = None
    ############################################################################
    # TODO: Execute o passo para frente, calculando as pontuações da categoria #
    # para a entrada. Armazene o resultado na variável "pontuacoes", que deve  #
    # ser um tensor de shape (N, C).                                           #
    ############################################################################
    # Substitua a comando "pass" pelo seu código
    h = X.mm(W1) + b1
    rep_oculta = h.clamp(min = 0)
    pontuacoes = rep_oculta.mm(W2) + b2
    ############################################################################
    #                             FIM DO SEU CODIGO                            #
    ############################################################################

    return pontuacoes, rep_oculta


def rn_frente_tras(params, X, y=None, reg=0.0):
    """
    Calcula a perda e os gradientes de uma rede neural totalmente conectada de duas 
    camadas. Ao implementar perda e gradiente, por favor, não se esqueça de dimensionar 
    as perdas/gradientes pelo tamanho do lote.

    Entrada: Os primeiros dois parâmetros (params, X) são iguais a rn_passo_para_frente
    - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
       um modelo. Deve ter as seguintes chaves com shape:
           W1: Pesos da primeira camada; tem shape (D, H)
           b1: Vieses da primeira camada; tem shape (H,)
           W2: Pesos da segunda camada; tem shape (H, C)
           b2: Vieses da segunda camada; tem shape (C,)
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.
    - y: Vetor de rótulos de treinamento. y[i] é o rótulo de X[i], e cada y[i] é
      um número inteiro no intervalo 0 <= y[i] <C. Este parâmetro é opcional; Se 
      ele não for informado, retornamos apenas pontuações e, se for informado,
      em vez disso, retornamos a perda e os gradientes.
    - reg: Força de regularização.

    Retorno:
    Se y for None, retorna um tensor de pontuações de shape (N, C) onde pontuacoes[i, c] 
    é a pontuação para a classe c na entrada X[i].

    Se y não for None, em vez disso, retorna uma tupla de:
    - perda: Perda (perda de dados e perda de regularização) para amostras deste lote 
      de treinamento.
    - grads: Dicionário mapeando nomes de parâmetros aos gradientes desses parâmetros
      com relação à função de perda; tem as mesmas chaves que self.params.    
    """
    # Descompacte as variáveis do dicionário de parâmetros
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    pontuacoes, h1 = rn_passo_para_frente(params, X)
    # Se os rótulos não forem fornecidos, então retorne, nada precisa ser feito
    if y is None:
      return pontuacoes

    # Calcular a perda
    perda = None
    ################################################################################
    # TODO: Calcule a perda com base nos resultados de rn_passo_para_frente. Isso  #
    # deve incluir a perda de dados e de regularização L2 para W1 e W2. Armazene o #
    # resultado na variável "perda", que deve ser um escalar. Use a perda do       # 
    # classificador Softmax. Ao implementar a regularização sobre W, por favor,    #
    # NÃO multiplique o termo de regularização por 1/2 (sem coeficiente). Se você  # 
    # não for cuidadoso aqui, é fácil encontrar instabilidade numérica (verifique  #
    # a estabilidade numérica em http://cs231n.github.io/linear-classify/).        #
    ################################################################################
    # Substitua a comando "pass" pelo seu código   
    max_elements, max_idxs = torch.max(pontuacoes,dim = 1, keepdim= True)
    exp_pontos = torch.exp(pontuacoes - max_elements)
    probs = exp_pontos/torch.sum(exp_pontos,dim = 1,keepdim=True)
    logprobs = -torch.log(probs[torch.arange(N),y])
    data_loss = torch.sum(logprobs)/N
    reg_loss = reg*torch.sum(W1*W1) + reg*torch.sum(W2*W2)
    perda = data_loss + reg_loss
    ################################################################################
    #                               FIM DO SEU CODIGO                              #
    ################################################################################

    # Passo para trás: calcular gradientes
    grads = {}
    ################################################################################
    # TODO: Execute o passo para trás, calculando as derivadas dos pesos e vieses. #
    # Armazene os resultados no dicionário "grads". Por exemplo, grads['W1'] deve  #
    # armazenar o gradiente em W1, e ser um tensor do mesmo tamanho                #
    ################################################################################
    # Substitua a comando "pass" pelo seu código
    dpontos = probs.clone()
    dpontos[torch.arange(N),y] -= 1
    dpontos /= N
    grads['W2'] = torch.mm(h1.t(),dpontos)
    grads['b2'] = torch.sum(dpontos, dim = 0, keepdim= True)

    dhidden = torch.mm(dpontos, W2.t())
    dhidden[h1 == 0] = 0
    grads['W1'] = torch.mm(X.t(),dhidden)
    grads['b1'] = torch.sum(dhidden, dim = 0, keepdim = True)

    grads['W1'] += 2*reg*W1
    grads['W2'] += 2*reg*W2
    ################################################################################
    #                               FIM DO SEU CODIGO                              #
    ################################################################################

    return perda, grads


def rn_treinar(params, func_perda, func_inferencia, X, y, X_val, y_val,
               taxa_de_aprendizado=1e-3, decaimento_taxa_de_aprendizado=0.95,
               reg=5e-6, num_iteracoes=100,
               tamanho_lote=200, verbose=False):
  """
  Treina essa rede neural usando a descida de gradiente estocástica.

  Entrada:
  - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
      um modelo. Deve ter as seguintes chaves com shape:
        W1: Pesos da primeira camada; tem shape (D, H)
        b1: Vieses da primeira camada; tem shape (H,)
        W2: Pesos da segunda camada; tem shape (H, C)
        b2: Vieses da segunda camada; tem shape (C,)
  - func_perda: Uma função de perda que calcula a perda e os gradientes.
    Recebe como entrada:
    - params: O mesmo que é fornecido para rn_treinar
    - X_lote: Um mini-lote de entradas de shape (B, D)
    - y_lote: Rótulos verdadeiros para X_loto
    - reg: O mesmo que é fornecido para rn_treinar
    E ele retorna uma tupla de:
      - perda: Escalar contendo a perda no mini-lote
      - grads: Dicionário mapeando nomes de parâmetros aos gradientes da perda
      com relação ao parâmetro correspondente.
  - func_inferencia: Função de inferência
  - X: Um tensor do PyTorch de shape (N, D) contendo dados de treinamento.
  - y: Um tensor do PyTorch de shape (N,) contendo rótulos de treinamento; 
      y[i] = c significa que X[i] tem rótulo c, onde 0 <= c < C.
  - X_val: Um tensor do PyTorch de shape (N_val, D) contendo dados de validação.
  - y_val: Um tensor do PyTorch de shape (N_val,) contendo rótulos de validação.
  - taxa_de_aprendizado: escalar indicando taxa de aprendizado para otimização.
  - decaimento_taxa_de_aprendizado: escalar indicando o fator usado para diminuir 
    a taxa de aprendizado após cada época.
  - reg: Escalar indicando a força de regularização.
  - num_iteracoes: Número de iterações a serem executadas durante a otimização.
  - tamanho_lote: Número de amostras de treinamento a serem usados ​​por etapa.
  - verbose: Booleano; se for verdadeiro imprime o progresso durante a otimização.

  Retorna: Um dicionário com estatísticas sobre o processo de treinamento.
  """
  num_entradas = X.shape[0]
  iteracoes_por_epoca = max(num_entradas // tamanho_lote, 1)

  # Use SGD para otimizar os parâmetros em self.model
  historico_perda = []
  historico_acc_treinamento = []
  historico_acc_validacao = []

  for it in range(num_iteracoes):
    # TODO: implemente a rotina amostrar_lote
    X_lote, y_lote = amostrar_lote(X, y, num_entradas, tamanho_lote)

    # Calcule a perda e os gradientes usando o mini-lote atual
    perda, grads = func_perda(params, X_lote, y=y_lote, reg=reg)
    historico_perda.append(perda.item())

    #######################################################################
    # TODO: Use os gradientes no dicionário "grads" para atualizar os     #
    # parâmetros da rede (armazenados no dicionário self.params) usando a #
    # descida de gradiente estocástica. Você precisará usar os gradientes #
    # armazenados no dicionário "grads" definido acima.                   #
    #######################################################################
    # Substitua a comando "pass" pelo seu código
    params['W1'] -= taxa_de_aprendizado * grads['W1']
    params['W2'] -= taxa_de_aprendizado * grads['W2']
    params['b1'] -= taxa_de_aprendizado * torch.flatten(grads['b1'])
    params['b2'] -= taxa_de_aprendizado * torch.flatten(grads['b2'])
    #######################################################################
    #                           FIM DO SEU CODIGO                         #
    #######################################################################

    if verbose and it % 100 == 0:
      print('iteração %d / %d: perda %f' % (it, num_iteracoes, perda.item()))

    # A cada época, verifique as acurácias de treinamento e de validação e 
    # reduza a taxa de aprendizado.
    if it % iteracoes_por_epoca == 0:
      # Verifique a acurácia
      y_pred_treinamento = func_inferencia(params, func_perda, X_lote)
      acc_treinamento = (y_pred_treinamento == y_lote).float().mean().item()
      y_pred_validacao = func_inferencia(params, func_perda, X_val)
      acc_validacao = (y_pred_validacao == y_val).float().mean().item()
      historico_acc_treinamento.append(acc_treinamento)
      historico_acc_validacao.append(acc_validacao)

      # Reduza a taxa de aprendizado
      taxa_de_aprendizado *= decaimento_taxa_de_aprendizado

  return {
    'historico_perda': historico_perda,
    'historico_acc_treinamento': historico_acc_treinamento,
    'historico_acc_validacao': historico_acc_validacao,
  }


def rn_inferir(params, func_perda, X):
  """
  Usa os pesos treinados desta rede de duas camadas para inferir rótulos para
  os dados. Para cada amostra de dados, prevemos pontuações para cada uma das C
  categorias e atribuímos cada amostra de dados à classe com a maior pontuação.

  Entrada:
    - params: Um dicionário de tensores do PyTorch que armazena os pesos de 
        um modelo. Deve ter as seguintes chaves com shape:
          W1: Pesos da primeira camada; tem shape (D, H)
          b1: Vieses da primeira camada; tem shape (H,)
          W2: Pesos da segunda camada; tem shape (H, C)
          b2: Vieses da segunda camada; tem shape (C,)
    - func_perda: Uma função de perda que calcula a perda e os gradientes.
     - X: Um tensor do PyTorch de shape (N, D) contendo N amostras D-dimensional 
       de dados para classificar.

  Retorno:
  - y_pred: Um tensor do PyTorch de shape (N,) contendo rótulos previstos para 
    cada um dos elementos de X. Para todo i, y_pred[i] = c significa que X[i] é 
    previsto ser da classe c, onde 0 <= c < C.    
  """
  y_pred = None

  ###########################################################################
  # TODO: Implemente esta função; deve ser MUITO simples!                   #
  ###########################################################################
  # Substitua a comando "pass" pelo seu código
  y_pred = torch.argmax(func_perda(params, X), 1)
  ###########################################################################
  #                           FIM DO SEU CODIGO                             #
  ###########################################################################

  return y_pred


def rn_params_busca_em_grade():
  """
  Retorna os hiperparâmetros candidatos para um modelo RedeDuasCamadas. 
  Você deve fornecer pelo menos dois parâmetros para cada um e o total de 
  combinações de busca em grade deve ser inferior a 256. Caso contrário, 
  levará muito tempo para treinar em tais combinações de hiperparâmetros.

  Retorno:
  - taxas_de_aprendizado: candidatos a taxa de aprendizado, por exemplo, 
                          [1e-3, 1e-2, ...]
  - tamanhos_oculta: tamanhos para a camada oculta, por exemplo, [8, 16, ...]
  - regs: candidatos a forças de regularização, por exemplo, [1e0, 1e1, ...]
  - decaimentos_taxa_de_aprendizado: candidatos à decaimento da taxa de 
                                     aprendizado, por exemplo, [1.0, 0.95, ...]    
  """
  taxas_de_aprendizado = []
  tamanhos_oculta = []
  regs = []
  decaimentos_taxa_de_aprendizado = []
  #############################################################################
  # TODO: Adicione suas próprias listas de hiperparâmetros. Isso deve ser     #
  # semelhante aos hiperparâmetros usados para o SVM, mas pode ser necessário #
  # selecionar diferentes hiperparâmetros para obter um bom desempenho com o  #
  # classificador softmax.                                                    #
  #############################################################################
  # Substitua a comando "pass" pelo seu código
  taxas_de_aprendizado = [0.3,0.2,0.1]
  tamanhos_oculta = [512,128,256]
  regs = [0.001,0.1,0.01]
  decaimentos_taxa_de_aprendizado = [0.95,0.99,0.995]
  #############################################################################
  #                            FIM DO SEU CODIGO                              #
  #############################################################################

  return taxas_de_aprendizado, tamanhos_oculta, regs, decaimentos_taxa_de_aprendizado


def encontrar_melhor_rede(dic_dados, func_params_busca):
  """
  Ajuste de hiperparâmetros usando o conjunto de validação.
  Armazene seu modelo RedeDuasCamadas mais bem treinado em melhor_rede, com o 
  valor de retorno da operação ".train()" em melhor_stat e a acurácia de validação 
  do melhor modelo treinado em melhor_acc_validacao. Seus hiperparâmetros devem 
  ser obtidos a partir de rn_params_busca_em_grade.    

  Entrada:
  - dic_dados (dicionário): Um dicionário que inclui
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      como as chaves para treinar um classificador
  - func_params_busca (função): Uma função que fornece os hiperparâmetros
                                (p.ex., rn_params_busca_em_grade) e retorna
                                (taxas_de_aprendizado, tamanhos_oculta,
                                regs, decaimentos_taxa_de_aprendizado)
                                Você deve obter os hiperparâmetros de
                                func_params_busca.

  Retorno:
  - melhor_rede (instância): uma instância de RedeDuasCamadas treinada com
                             (['X_train', 'y_train'], tamanho_lote, 
                             taxa_de_aprendizado, decaimento_taxa_de_aprendizado, 
                             regs) por num_iteracoes vezes.
  - melhor_stat (dicionário): valor de retorno da operação "melhor_rede.treinar()"
  - melhor_acc_validacao (float): acurácia de validação da melhor_rede
  """

  melhor_rede = None
  melhor_stat = None
  melhor_acc_validacao = 0.0

  #########################################################################
  # TODO: Ajuste hiperparâmetros usando o conjunto de validação. Armazene #
  # seu modelo mais bem treinado em melhor_rede.                          #
  #                                                                       #
  # Para ajudar a depurar sua rede, pode ser útil usar visualizações      #
  # semelhantes às que usamos acima; essas visualizações terão diferenças #
  # qualitativas significativas das que vimos para uma rede mal ajustada. #
  #                                                                       #
  # Ajustar hiperparâmetros manualmente pode ser divertido, mas você pode #
  # achar útil escrever um código para varrer as possíveis combinações de # 
  # hiperparâmetros automaticamente.                                      #
  #########################################################################
  # Substitua a comando "pass" pelo seu código  
  taxas_de_aprendizado, tamanhos_oculta, regs, decaimentos_taxa_de_aprendizado = func_params_busca()
  tam_xt = dic_dados['X_train'].shape[1]
  tam_yt = 10
  for i in taxas_de_aprendizado:
    for j in tamanhos_oculta:
      for k in regs:
        for t in decaimentos_taxa_de_aprendizado:
          rede = RedeDuasCamadas(tam_xt,j,tam_yt)
          hist_rede = rede.treinar(dic_dados['X_train'],dic_dados['y_train'], dic_dados['X_val'],dic_dados['y_val'],i,t,k,num_iteracoes= 2500)
          if(hist_rede['historico_acc_validacao'][-1] > melhor_acc_validacao):
            melhor_rede = rede
            melhor_acc_validacao = hist_rede['historico_acc_validacao'][-1]
            melhor_stat = hist_rede
  #########################################################################
  #                          FIM DO SEU CODIGO                            #
  #########################################################################

  return melhor_rede, melhor_stat, melhor_acc_validacao
