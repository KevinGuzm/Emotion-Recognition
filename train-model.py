import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

# Carga de datos
data_file = "data.txt"
data = np.loadtxt(data_file)

X = data[:, :-1]
y = data[:, -1]

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, shuffle=True,
                                                    stratify=y)

# Definición del espacio de hiperparámetros
param_space = {
    'n_estimators': Integer(50, 200),            # Número de árboles
    'max_depth': Integer(10, 50),                # Profundidad máxima
    'min_samples_split': Integer(2, 10),         # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': Integer(1, 10),          # Mínimo de muestras en una hoja
    'max_features': Categorical(['sqrt', 'log2'])  # Número de características para cada división
}

# Inicialización de RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Configuración de BayesSearchCV para optimización bayesiana
bayes_search = BayesSearchCV(
    rf_classifier,
    search_spaces=param_space,
    n_iter=100,                # Número de iteraciones de búsqueda
    cv=10,                     # Validación cruzada de 5 pliegues
    scoring='accuracy',       # Métrica a optimizar
    random_state=42,
    n_jobs=-1,                # Usa todos los núcleos disponibles
    verbose=2
)

# Ajuste del modelo con la búsqueda de hiperparámetros
bayes_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_rf_classifier = bayes_search.best_estimator_

# Evaluación del mejor modelo
y_pred = best_rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best Parameters: {bayes_search.best_params_}")
print(f"Best Cross-Validated Accuracy: {bayes_search.best_score_ * 100:.2f}%")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Guardado del modelo optimizado
with open('./optimized_model', 'wb') as f:
    pickle.dump(best_rf_classifier, f)
