import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# 1. Geração da base tabular anonimizada calibrada às regras de RPPS (N = 2.500)
np.random.seed(42)
N = 2500

sexo = np.random.choice([0, 1], size=N, p=[0.42, 0.58])  # 0: Masculino, 1: Feminino
regra_pleiteada = np.random.choice(
    ['voluntaria', 'transicao_pontos', 'especial'], size=N, p=[0.45, 0.40, 0.15]
)
idade_der = np.clip(np.round(np.random.normal(loc=59.5, scale=4.5, size=N), 1), 45.0, 74.0)
tempo_contrib = np.random.randint(8500, 14500, size=N)
tempo_serv_pub = np.minimum(tempo_contrib, np.random.randint(3650, 13000, size=N))
tempo_cargo = np.minimum(tempo_serv_pub, np.random.randint(1000, 11000, size=N))
possui_ctc = np.random.choice([0, 1], size=N, p=[0.65, 0.35])
pendencia_cad = np.random.choice([0, 1], size=N, p=[0.82, 0.18])

df = pd.DataFrame({
    'idade_der': idade_der,
    'tempo_contrib': tempo_contrib,
    'tempo_serv_pub': tempo_serv_pub,
    'tempo_cargo': tempo_cargo,
    'sexo': sexo,
    'regra_pleiteada': regra_pleiteada,
    'possui_ctc': possui_ctc,
    'pendencia_cad': pendencia_cad
})

# 2. Engenharia de Atributos (EC 103/2019)
df['razao_cargo'] = df['tempo_cargo'] / df['tempo_serv_pub']
df['pontos_ec103'] = df['idade_der'] + (df['tempo_contrib'] / 365.25)

# Definição do desfecho (0 = deferido, 1 = indeferido)
idade_min = np.where(df['sexo'] == 0, 62.0, 57.0)
contrib_min = np.where(df['sexo'] == 0, 12775, 10950)
pontos_min = np.where(df['sexo'] == 0, 99.0, 89.0)

cond_voluntaria = (
    (df['idade_der'] >= idade_min) & (df['tempo_contrib'] >= contrib_min) &
    (df['tempo_serv_pub'] >= 7300) & (df['tempo_cargo'] >= 1825)
)
cond_pontos = (
    (df['pontos_ec103'] >= pontos_min) & (df['tempo_contrib'] >= contrib_min) &
    (df['tempo_serv_pub'] >= 7300) & (df['tempo_cargo'] >= 1825)
)
cond_especial = (
    (df['idade_der'] >= 55.0) & (df['tempo_contrib'] >= 9125) &
    (df['tempo_serv_pub'] >= 3650) & (df['tempo_cargo'] >= 1825)
)

elegivel_regra = np.where(
    df['regra_pleiteada'] == 'voluntaria', cond_voluntaria,
    np.where(df['regra_pleiteada'] == 'transicao_pontos', cond_pontos, cond_especial)
)

indeferido = (~elegivel_regra) | ((df['pendencia_cad'] == 1) & (np.random.rand(N) < 0.78))
ruido = np.random.rand(N) < 0.04
df['y'] = np.where(ruido, ~indeferido, indeferido).astype(int)

# 3. Particionamento estratificado (80% treino / 20% teste)
X = df.drop(columns=['y'])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cat_cols = ['regra_pleiteada']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

# 4. Treinamento com Validação Cruzada Estratificada (5-fold) e Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
neg, pos = np.bincount(y_train)
scale_pos = neg / pos

xgb_estimator = (
    XGBClassifier(random_state=42, eval_metric='logloss', colsample_bytree=0.8, scale_pos_weight=scale_pos)
    if HAS_XGB else GradientBoostingClassifier(random_state=42)
)

models = {
    'CART': (
        DecisionTreeClassifier(random_state=42),
        {'clf__ccp_alpha': [0.0, 0.005, 0.01], 'clf__max_depth': [4, 6, 8]}
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'clf__n_estimators': [100, 200], 'clf__max_depth': [6, 10, None]}
    ),
    'XGBoost': (
        xgb_estimator,
        {'clf__learning_rate': [0.03, 0.05, 0.1], 'clf__max_depth': [3, 5, 6]}
    )
}

for name, (clf, grid) in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring='f1', n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    print(f"=== {name} ===")
    print(f"Melhores parâmetros: {gs.best_params_}")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f} | Precisão: {precision_score(y_test, y_pred):.4f} | "
          f"Recall: {recall_score(y_test, y_pred):.4f} | F1-Score: {f1_score(y_test, y_pred):.4f} | "
          f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}\n")
