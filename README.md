# <img width="32" height="32" alt="deep-learning2" src="https://github.com/user-attachments/assets/5ff88a35-02ab-4b23-95a2-250b66bb75f0" /> AutoencoderVanila - Detecting SQL Injection and XSS (Cross-Site-Scripting)
![GitHub stars](https://img.shields.io/github/stars/mackcoder/ML-models-AutoencoderVanila-in-Python.-?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/mackcoder/ML-models-AutoencoderVanila-in-Python.-?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/mackcoder/ML-models-AutoencoderVanila-in-Python.-?style=for-the-badge)

## 📝Desciption
<div align="justify">
  This study aims to improve website defense systems through unsupervised machine
learning methods to detect SQL Injection (SQLi) and Cross-Site Scripting (XSS) attacks.
Using Python and artificial intelligence libraries, an autoencoder model was developed and
trained to identify anomalous patterns in web traffic. Publicly available datasets were applied
for training and validation. The results indicated high efficiency in detecting SQL Injection,
with strong recall and low false negatives, although with high false positive rates for XSS
detection. It is concluded that the proposed approach is promising but requires further
optimization before being used in production environments
</div>

## 🌟 Highlights
  - 👾SQL Injection and XSS (Cross-Site-Scripting)
  - 📂Training with Public Datasets 
  - 🛠️Implementation in Python with TensorFlow
  - 🧠Unsupervised Machine Leaning with Autoencoder
  - ❗Detection tests and results

# Getting Started
## 🥣 Prerequisites
  All following experiments were conducted using the **GoogleColab** enviroment.
   
  ## 📚 Libraries Used
  - numpy - For numerical operations and array manipulation
  - pandas - For data loading and preprocessing
  - scikit-learn - For model evaluation, data splitting and scaling
  - tensorflow - For building and training the Autoencoder model
  - keras - High-level API for defining neural network layers
  - classification_report & confusion_matrix - For performance metrics
    
  ## 🗃️ Data Acquisition and Preprocessing
  ### 📑 Datasets
  - [Download SQL Injection Attack for Training (D1).csv](https://zenodo.org/records/6906893)
  - [Download SQL Injection Attack for Test (D2).csv](https://zenodo.org/records/6907252)
  - [Download XSSTraining.csv](https://github.com/fmereani/Cross-Site-Scripting-XSS/blob/master/XSSDataSets/XSSTesting.csv)
  - [Download XSSTesting.csv](https://github.com/fmereani/Cross-Site-Scripting-XSS/blob/master/XSSDataSets/XSSTraining.csv)
  ### Uploading Datasets in Google Colab:
  - Drag and drop the files into the Files pane on the left sidebar;
  **OR**
  - Upload them programmatically using the following code:
  ---
  ```python
  from google.colab import files
  uploaded = files.upload()
  ```
  ---
  ### Preprocessing data
  > [!IMPORTANT]
  > **For SQL Injection**
  ##
  Explanation: 
  ---
  ```python
  dfxss = pd.read_csv('SLQ Injection Attack for training (D1) (1).csv')
  y = dfxss['Label']
  X = dfxss.select_dtypes(include = ['number']).drop('Label', axis = 1)
  X_treino, X_teste, y_treino, y_teste = train_test_split( X, y, test_size=0.2,                   random_state=42, stratify=y)
  #2 Normalização dos dados
  escala = StandardScaler()
  X_treino_normal = X_treino[y_treino == 0]  #Filtra apenas os dados normais para treinar o autoencoder
  X_treino_normal_escala = escala.fit_transform(X_treino_normal)  #Ajusta e transforma os dados normais
  X_teste_escala = escala.transform(X_teste)  #Aplica a mesma transformação ao conjunto de teste
  ```
  ---
  
  ##
  > [!IMPORTANT]
  > **For XSS**
  ##
  Explanation: 
  ---
  ```python
  #1 Carrega o dataset de treino
  df_treino = pd.read_csv("XSSTraining.csv")
  
  # Separa os dados em variáveis independentes (X) e a variável alvo (y):
  X = df_treino.drop("Class", axis=1)  # Remove a coluna "Class" para usar como entrada
  y = df_treino["Class"].apply(lambda x: 1 if x == "Malicious" else 0)  # Converte rótulo para binário: 1 = ataque, 0 = normal
  
  #2 Divide os dados em treino e teste, mantendo a proporção das classes
  X_treino, X_teste, y_treino, y_teste = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y  # 20% para teste, estratificado por classe
  )
  
  #3 Normaliza apenas os dados normais do conjunto de treino
  escala = StandardScaler()  # Inicializa o normalizador
  X_treino_normal = X_treino[y_treino == 0]  # Filtra apenas os dados normais
  X_treino_normal_escala = escala.fit_transform(X_treino_normal)  # Ajusta e transforma os      dados normais
  X_teste_escala = escala.transform(X_teste)  # Aplica a mesma transformação ao conjunto de teste

  ```
  ---
  ##
  
  ## 🤖Implementing Autoencoder 
   <img width="450" height="361" alt="image" src="https://github.com/user-attachments/assets/c78d250d-0e9c-4aa0-ab4c-e79a2c1423b3" />
   
   > [!IMPORTANT]
  > **Model for SQL Injection**
  ##
  Explanation: 
  ---
  ```python
  #3 Criação da arquitetura do Autoencoder
input_dim = X_treino.shape[1]  #Define o número de atributos de entrada
input_layer = Input(shape=(input_dim,))  #Camada de entrada

# Camadas de codificação (reduzem a dimensionalidade)
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)

# Camadas de decodificação (reconstrução dos dados)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='relu')(decoded)  # Camada final com mesma dimensão da entrada

# Cria o modelo Autoencoder e compila com otimizador Adam e função de perda MSE
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

#4 Treinamento do Autoencoder com os dados normais
autoencoder.fit(
    X_treino_normal_escala, X_treino_normal_escala,
    epochs=90,  #Número de ciclos
    batch_size=32,  #Tamanho do lote
    shuffle=True,  #Embaralha os dados a cada época
    validation_split=0.1,  #10% dos dados para validação
    verbose=1  #Exibe progresso do treinamento
)

#5 Reconstrução dos dados de teste e cálculo do erro de reconstrução
X_recalculo = autoencoder.predict(X_teste_escala)  #Reconstrói os dados de teste
error = np.mean(np.square(X_teste_escala - X_recalculo), axis=1)  #Calcula o erro por amostra

# Reconstrução dos dados normais de treino e cálculo do erro
reco_treino = autoencoder.predict(X_treino_normal_escala)
mse_train = np.mean(np.square(X_treino_normal_escala - reco_treino), axis=1)

# 6. Loop para encontrar o melhor threshold (limiar de detecção)
porcentagens = range(70, 100, 2)  # Testa thresholds entre 70 e 98
melhor_recall = 0  #Inicializa o melhor recall
melhor_threshold = 0  #Inicializa o melhor threshold
melhor_resultado = {}  #Dicionário para armazenar o melhor resultado

y_teste_numeric = y_teste.values  #Converte os rótulos para array NumPy

# Loop para testar diferentes thresholds:
for x in porcentagens:
    threshold = np.percentile(mse_train, x)  #Define o threshold com base no erro dos dados normais
    prev_loop = [1 if i > threshold else 0 for i in error]  #Classifica como ataque se erro > threshold

    #Gera relatório de classificação para o threshold atual:
    relatorio = classification_report(
        y_teste_numeric, prev_loop,
        target_names=["Normal", "Ataque"],
        output_dict=True
    )
    recall_ataque = relatorio["Ataque"]["recall"]  #Extrai o recall da classe "Ataque"

    # Atualiza o melhor resultado se o recall for maior:
    if recall_ataque > melhor_recall:
        melhor_recall = recall_ataque
        melhor_threshold = threshold
        melhor_resultado = {
            "percentil": x,
            "precision": relatorio["Ataque"]["precision"],
            "recall": recall_ataque,
            "f1": relatorio["Ataque"]["f1-score"],
            "matriz_confusao": confusion_matrix(y_teste_numeric, prev_loop)
        }

  ```
  ---
  ##

   > [!IMPORTANT]
  > **🏋️‍♂️Model training for XSS**
  ##
  Explanation: 
  ---
  ```python
  #4 Define a arquitetura do Autoencoder
input_dim = X.shape[1]  # Número de atributos de entrada
input_layer = Input(shape=(input_dim,))  # Camada de entrada

# Camadas de codificação
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)

# Camadas de decodificação
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='relu')(decoded)  #Reconstrói a entrada

# Cria e compila o modelo Autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')  #Usa o erro como função de perda

#5 Treina o Autoencoder apenas com dados normais
autoencoder.fit(
    X_treino_normal_escala, X_treino_normal_escala,  #Entrada e saída são iguais
    epochs=90, batch_size=32, shuffle=True,  #Treinamento por 90 épocas/ciclos
    validation_split=0.1, verbose=1  # Usa 10% dos dados normais para validação
)

#6 Calcula o erro de reconstrução no conjunto de teste
X_recalculo = autoencoder.predict(X_teste_escala)  #Reconstrói os dados de teste
error = np.mean(np.square(X_teste_escala - X_recalculo), axis=1)  #Erro de reconstrução por amostra

# Calcula o erro de reconstrução nos dados normais de treino
reco_treino = autoencoder.predict(X_treino_normal_escala)
mse_train = np.mean(np.square(X_treino_normal_escala - reco_treino), axis=1)

#7 Busca o melhor threshold para detectar ataques
porcentagens = range(70, 100, 2)  #Testa thresholds entre 70 e 98
melhor_recall = 0
melhor_threshold = 0
melhor_resultado = {}

y_teste_numeric = y_teste.values  #Converte para array NumPy

# Loop para testar diferentes thresholds
for x in porcentagens:
    threshold = np.percentile(mse_train, x)  #Define threshold com base no erro dos dados normais
    prev_loop = [1 if i > threshold else 0 for i in error]  #Classifica como ataque se erro > threshold

    # Gera relatório de classificação
    relatorio = classification_report(
        y_teste_numeric, prev_loop,
        target_names=["Normal", "Ataque"],
        output_dict=True
    )
    recall_ataque = relatorio["Ataque"]["recall"]  #Extrai o recall da classe "Ataque"

    # Atualiza o melhor resultado se o recall for maior
    if recall_ataque > melhor_recall:
        melhor_recall = recall_ataque
        melhor_threshold = threshold
        melhor_resultado = {
            "percentil": x,
            "precision": relatorio["Ataque"]["precision"],
            "recall": recall_ataque,
            "f1": relatorio["Ataque"]["f1-score"],
            "matriz_confusao": confusion_matrix(y_teste_numeric, prev_loop)
        }

  ```
  ---
  
  ##
  

   > [!IMPORTANT]
  > **🏋️‍♂️Model testing for XSS**
  ##
  Explanation: 
  ---
  ```python
  df_xss = pd.read_csv("XSSTesting.csv")
  y_xss = df_xss["Class"].apply(lambda x: 1 if x == "Malicious" else 0)  #Converte rótulo para binário
  X_xss = df_xss.drop("Class", axis=1)  #Remove a coluna de classe
  
  X_xss_escala = escala.transform(X_xss)  #Normaliza os dados com o mesmo scaler
  X_xss_reconstruido = autoencoder.predict(X_xss_escala)  #Reconstrói os dados
  erro_xss = np.mean(np.square(X_xss_escala - X_xss_reconstruido), axis=1)  #Calcula erro de reconstrução
  
  prev_xss = [1 if e > melhor_threshold else 0 for e in erro_xss]  #Classifica com base no threshold ideal
  ```
  ---
  
  ##

# 🧪 Running the tests








