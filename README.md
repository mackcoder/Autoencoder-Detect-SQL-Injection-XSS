# <img width="32" height="32" alt="deep-learning2" src="https://github.com/user-attachments/assets/5ff88a35-02ab-4b23-95a2-250b66bb75f0" /> AutoencoderVanila - Detecting SQL Injection and XSS (Cross-Site-Scripting)
![GitHub stars](https://img.shields.io/github/stars/mackcoder/ML-models-AutoencoderVanila-in-Python.-?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/mackcoder/ML-models-AutoencoderVanila-in-Python.-?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/mackcoder/ML-models-AutoencoderVanila-in-Python.-?style=for-the-badge)

## 📝Description
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
 
   ## 📝 Explanation:
  <div align="justify">
  In this section, the CSV file containing the dataset (SQL Injection) is loaded. The target variable (y) is extracted from the 'Label' column, while the feature matrix (X) is composed of all numerical columns except 'Label'. The dataset is then split into training and testing sets using train_test_split, with 80% of the data allocated for training and 20% for testing. The stratify=y parameter ensures that the class distribution remains consistent across both sets.
Next, data normalization is performed using StandardScaler. Only the normal samples (where y_treino == 0) are used to train the autoencoder, so the scaler is fitted exclusively on these. The same scaling transformation is then applied to the entire test set to ensure consistency.
  </div>
  
  ##
   > [!NOTE]
   > **For SQL INJECTION**
  ##
  
  ---
  ```python
  dfxss = pd.read_csv('SLQ Injection Attack for training (D1) (1).csv')
  y = dfxss['Label']
  X = dfxss.select_dtypes(include = ['number']).drop('Label', axis = 1)
  X_treino, X_teste, y_treino, y_teste = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
  #2 Normalização dos dados
  escala = StandardScaler()
  X_treino_normal = X_treino[y_treino == 0]  #Filtra apenas os dados normais para treinar o autoencoder
  X_treino_normal_escala = escala.fit_transform(X_treino_normal)  #Ajusta e transforma os dados normais
  X_teste_escala = escala.transform(X_teste)  #Aplica a mesma transformação ao conjunto de teste
  ```
  ---
  
  ## 📝 Explanation: 
  <div align="justify">
  The XSS dataset underwent the same preprocessing steps used for the SQL Injection data. The only difference was in how the feature matrix (X) was extracted from the 'Label' collumn.
  </div>  
  
  ##
  
  > [!NOTE]
  > **For XSS**

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
  X_treino_normal_escala = escala.fit_transform(X_treino_normal)  # Ajusta e transforma os dados normais
  X_teste_escala = escala.transform(X_teste)  # Aplica a mesma transformação ao conjunto de teste

  ```

  ##  
  ## 🛠️Implementing Autoencoder 
   <img width="450" height="361" alt="image" src="https://github.com/user-attachments/assets/c78d250d-0e9c-4aa0-ab4c-e79a2c1423b3" />
   
   > [!IMPORTANT]
  > **Model for SQL Injection**
  ## 📝 Explanation:
  ### 1. 📐 Define Input Dimensions + Compression and decompression 
  - Determine the number of input features
  - Initialize the input layer for neural network
  - Compresses the input data into a lower-dimensional representation (encoded)
  - Helps the model learn the most relevant patterns from normal data (encoded)
  - The algorithym reconstructs the original input from the compressed representation by itself (decoded)
  - The final layer matches the original input shape (decoded)
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
```
  ### 2.🏋️ Model training
  - Building model
  - Normal is inserted in the autoencoder training
  - Adam optimizer and MSE (Mean Squared Error) used
  - 90 cycles (epochs) done to learn important patterns
  ```python
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
```
### 3.🏋️ Reconstruct model + Reference Error
  - Reconstruct test data
  - Calculate reconstruction error for each sample
  - Measures baseline error from normal training data
  - Used to define detection threshold
```python
#5 Reconstrução dos dados de teste e cálculo do erro de reconstrução
X_recalculo = autoencoder.predict(X_teste_escala)  #Reconstrói os dados de teste
error = np.mean(np.square(X_teste_escala - X_recalculo), axis=1)  #Calcula o erro por amostra

# Reconstrução dos dados normais de treino e cálculo do erro
reco_treino = autoencoder.predict(X_treino_normal_escala)
mse_train = np.mean(np.square(X_treino_normal_escala - reco_treino), axis=1)
```
### 4.🔬 Threshold optimization loop + Classification Report + Recall optimization
  - Loop created to identify the optimal threshold and recall
  - Generating classification report -> compares the model's reconstruction-bases predictions against the true labels
```python
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


   > [!IMPORTANT]
  > **🏋️‍♂️Model training for XSS**
  ## 📝 Explanation: 
The same training procedure used for the SQL Injection detection model was applied to the XSS (Cross-Site Scripting) detection model.  
This includes the autoencoder architecture, preprocessing steps, training strategy, and threshold optimization for anomaly classification.

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
  ## 📝 Explanation: Now the Dataset with testing data is loaded to test the model's capabilities to detect XSS patterns
  - Loads XSS Dataset into a DataFrame
  - Converts the "Class" column into binary labels: 1 for malicious, 0 for normal
  - Separates the feature set by removing the label column
  - Applies the same scaler used during training to ensure consistent data distribution
  - Uses the trained autoencoder to reconstruct the input data
  - Computes the mean squared error for each sample to quantify how well the model reconstructed it
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

## 👾 Model trained for SQL INJECTION:                            

|   Classe   | Precisão  | Recall | F1-Score |
|------------|---------- |--------|----------|
|  NORMAL    |   0.92    |  0.81  |   0.86   |
|  ATAQUE    |   0.60    |  0.80  |   0.69   |
|------------|-----------|--------|----------|
| ACCURACY   |           |        |   0.91   |
| MACRO AVG  |   0.91    |  0.92  |   0.91   |
| WEIGHT AVG |   0.92    |  0.92  |   0.91   |

## 👾 Model trained for CROSS-SITE SCRIPTING:

|   Classe   | Precisão  | Recall | F1-Score |
|------------|---------- |--------|----------|
|  NORMAL    |   1.00    |  0.84  |   0.91   |
|  ATAQUE    |   0.82    |  1.00  |   0.90   |
|------------|-----------|--------|----------|
| ACCURACY   |           |        |   0.91   |
| MACRO AVG  |   0.91    |  0.92  |   0.91   |
| WEIGHT AVG |   0.92    |  0.91  |   0.91   |

## Comparison:

| Attack Type       | Metric | Precision | Recall | F1-Score | Accuracy |
| ----------------- | ------ | --------- | ------ | -------- | -------- |
| **SQL Injection** | Normal | 0.92      | 0.81   | 0.86     | **0.91** |
|                   | Attack | 0.60      | 0.80   | 0.69     |          |
| **XSS**           | Normal | 1.00      | 0.84   | 0.91     | **0.91** |
|                   | Attack | 0.82      | 1.00   | 0.90     |          |

## 📊 Graphics


<img width="2884" height="1467" alt="SQLINjection_graphic" src="https://github.com/user-attachments/assets/c6511cfb-e16c-4022-a7ee-22f12a1eed65" />

---
  

<img width="2884" height="1452" alt="XSS_graphic" src="https://github.com/user-attachments/assets/fa4df1a7-feef-4a17-8c8a-d8c73b977a71" />

## ✍️ Final Considerations
This project demonstrates the potential of **unsupervised anomaly detection** with Autoencoders in cybersecurity applications. The method showed strong recall for attack detection, particularly for **SQL Injection**, but further improvements are needed to reduce false positives in **XSS** detection.

🔮 Future Work May Include: 
- Enhancing the pre-processing method to improve data quality and model performance

- Experimenting with different neuron configurations to obtain more robust and interpretable results

- Validating the model using real traffic data to observe its behavior, assess accuracy, and measure execution time

- Applying ROC curve analysis to visually identify the optimal threshold, replacing the current method with a more time-efficient approach

- Implementing adaptive monitoring techniques to continuously track performance metrics and adjust parameters as needed

## 👤 Author
- Naoto Ushizaki
- Email: 

## 👥 Contributor
- Prof. Dr. Rodrigo Cardoso Silva - Academic advisor and project supervisor 
  
## 📚 References
- MAC, Hieu; TRUONG, Dung; NGUYEN, Lam; NGUYEN, Hoa; TRAN, Hai Anh; TRAN, Duc. Detecting Attacks on Web Applications using Autoencoder. In: The Ninth International Symposium on Information and Communication Technology (SoICT 2018), Danang City, Viet Nam, 6 dez. 2018. Disponível em: https://admin.tvrijakartanews.com/uploads/Detecting_Attackson_Web_Applicationsusing_Autoencoder_dab6406334.pdf. Acesso em: 14 ago. 2025.
- AUGUSTINE, Nwabudike.; SULTAN, A.B.M.; OSMAN, M.H.; SHARIF, K.Y. Application of Artificial Intelligence in detecting SQL Injection Attacks. Internacional Journal On Informatics Visualization. 22 out. 2024. Disponível em: https://joiv.org/index.php/joiv/article/download/3631/1136. Acesso em: 14 ago. 2025.
- HOPHR. How to implement and scale up autoencoders in TensorFlow for anomaly detection or data generation tasks. HopHR, s.d. Disponível em: https://www.hophr.com/tutorial-page/implement-and-scale-up-autoencoders-in-tensorflow-for-anomaly-detection-or-data-generation-tasks. Acesso em: 03 ago. 2025.
- CAMPAZAS, A.; CRESPO, I. SQL Injection Attack for Training (D1). Zenodo, 26 jul. 2022. Disponível em: https://zenodo.org/records/6906893. Acesso em: 19 jul. 2025.
- CAMPAZAS, A.; CRESPO, I. SQL Injection Attack for Test (D2). Netflow. Zenodo, 26 jul. 2022. Disponível em: https://zenodo.org/records/6907252. Acesso em: 19 jul. 2025.
- MEREANI, F. A.; HOWE, J. M. (2018). Detecting Cross-Site Scripting Attacks Using Machine Learning. In Advanced Machine Learning Technologies and Applications, volume 723 of AISC, pages 200–210. Springer. Disponível em: https://github.com/fmereani/Cross-Site-Scripting-XSS/tree/master/XSSDataSets. Acesso em: 28 jul. 2025.
- LEHR, J. et al. Supervised learning vs. unsupervised learning: a comparison for optical inspection applications in quality control. IOP Conference Series: Materials Science and Engineering, v. 1140, 2021. Disponível em: https://iopscience.iop.org/article/10.1088/1757-899X/1140/1/012049/pdf. Acesso em: 14 fev. 2024.









