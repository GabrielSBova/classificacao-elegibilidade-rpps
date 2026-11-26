# Classificação Supervisionada de Elegibilidade Previdenciária em RPPS

Repositório oficial com o código-fonte utilizado no artigo científico **"Classificação Supervisionada de Elegibilidade Previdenciária em Regimes Próprios de Previdência Social"**, desenvolvido na disciplina de **Inteligência Artificial I** — Engenharia da Computação, **Centro Universitário Hermínio Ometto (FHO)**.

**Autor:** Gabriel Santim Bova

---

## Objetivo
Avaliar e comparar o desempenho de algoritmos de aprendizado de máquina supervisionado baseados em árvores de decisão (**CART**, **Random Forest** e **XGBoost**) na predição de elegibilidade e detecção preventiva de risco de indeferimento em requerimentos de aposentadoria de Regimes Próprios de Previdência Social (RPPS).

## Estrutura do Repositório
* `classificacao_rpps.py`: Script completo contendo a geração da base tabular anonimizada calibrada às regras previdenciárias (EC 103/2019), engenharia de atributos, divisão estratificada (80/20), validação cruzada estratificada (*5-fold*) com busca em grade (*Grid Search*) e avaliação das métricas no conjunto de teste.

## Requisitos e Execução
1. Instale as dependências:
```bash
pip install numpy pandas scikit-learn xgboost
1. Instale as dependências:
```bash
pip install numpy pandas scikit-learn xgboost
