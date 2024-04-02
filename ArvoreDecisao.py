import pandas as pd
import numpy as np

# Carregar os dados
data = pd.read_csv('/mnt/data/IA_-_TP01._-_toughestsport.csv.csv')

# Corrigir os dados conforme necessário (baseado nas correções anteriores)
data['END'] = pd.to_numeric(data['END'], errors='coerce')

# Preparar os dados
X = data.drop(columns=['index', 'SPORT', 'TOTAL', 'RANK']).to_numpy()  # Recursos
y = data['TOTAL'].to_numpy()  # Variável alvo

# Dividir os dados manualmente (uma abordagem simples sem validação cruzada)
# Esta é uma divisão simples para demonstração. Em prática, você usaria algo como train_test_split.
split_index = int(X.shape[0] * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Construir a árvore
tree = build_tree(X_train, y_train, max_depth=5)  # Limitando a profundidade para simplificar

# Fazer previsões
predictions = predict(tree, X_test)

# Avaliar o modelo (usando MSE, por exemplo)
mse_value = mse(y_test, np.array(predictions))
print(f"MSE: {mse_value}")