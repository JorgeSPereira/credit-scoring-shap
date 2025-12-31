# Credit Scoring com Interpretabilidade (SHAP)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.43+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> Modelo de Credit Scoring com foco em **interpretabilidade** utilizando SHAP, **API REST** para inferência, **MLflow** para tracking de experimentos e **análise de fairness** para detecção de viés.

---

## Destaques do Projeto

| Feature | Descrição |
|---------|-----------|
| **Optuna** | Otimização Bayesiana de hiperparâmetros |
| **SHAP** | Interpretabilidade global e local |
| **FastAPI** | API REST para predições em tempo real |
| **MLflow** | Tracking de experimentos e modelos |
| **Fairlearn** | Análise de viés e equidade |
| **Pandera** | Validação de dados |
| **CI/CD** | GitHub Actions para testes automatizados |

---

## Contexto de Negócio

A concessão de crédito é uma das decisões mais críticas em instituições financeiras. Um modelo de **Credit Scoring** precisa não apenas ser preciso, mas também **explicável** para:

- **Compliance regulatório** (LGPD, BACEN)
- **Transparência** com clientes (direito à explicação)
- **Validação de negócio** (features fazem sentido?)
- **Detecção de viés** em variáveis sensíveis

---

## Resultados

| Modelo | ROC-AUC | Precision | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| Logistic Regression | **0.806** | 0.537 | 0.720 | 0.658 |
| XGBoost (Optuna) | 0.781 | **0.560** | 0.589 | 0.573 |
| LightGBM (Optuna) | 0.782 | 0.545 | 0.600 | 0.581 |

---

## Foundation Models Comparison

Este projeto também explora **Foundation Models** para dados tabulares, comparando performance vs interpretabilidade.

### Modelos Avaliados

| Modelo | Tipo | Performance | Interpretabilidade |
|--------|------|-------------|-------------------|
| **XGBoost + SHAP** | Gradient Boosting | Boa | Excelente |
| **TabPFN** | Transformer pre-treinado | Excelente | Limitada |
| **MITRA** (AutoGluon) | Foundation Model | State-of-the-art | Limitada |

### Trade-off: Performance vs Explicabilidade

> **Importante**: Em credit scoring, a **explicabilidade é requisito regulatório** (LGPD, BACEN).

| Cenário | Modelo Recomendado |
|---------|-------------------|
| Produção (regulado) | XGBoost + SHAP |
| Prototipo/PoC | TabPFN/MITRA |
| Benchmark interno | Comparar todos |

Para detalhes, veja: `notebooks/05_foundation_models.ipynb`



## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/credit-scoring-shap.git
cd credit-scoring-shap

# Instale as dependências
poetry install

# Ative o ambiente
poetry shell
```

---

## Uso

### Treinar Modelo

```bash
# Com otimização Optuna (recomendado)
make train

# Treinamento rápido (sem Optuna)
make train-quick
```

### API REST

```bash
# Iniciar servidor
make api

# Testar endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"duration": 24, "credit_amount": 5000, ...}'
```

Acesse a documentação interativa em: `http://localhost:8000/docs`

### MLflow Tracking

```bash
# Iniciar UI do MLflow
make mlflow

# Acessar em: http://localhost:5000
```

### Análise de Fairness

```bash
# Executar notebook de fairness
make fairness
```

### Notebooks

```bash
poetry run jupyter notebook

# Ordem recomendada:
# 1. 01_eda.ipynb - Análise exploratória
# 2. 02_modeling.ipynb - Modelagem com Optuna
# 3. 03_interpretability.ipynb - Análise SHAP
# 4. 04_fairness.ipynb - Detecção de viés
# 5. 05_foundation_models.ipynb - TabPFN/MITRA
```

---

## Estrutura do Projeto

```
credit-scoring-shap/
├── .github/workflows/      # CI/CD
│   └── ci.yml
├── src/credit_scoring/
│   ├── api/                # FastAPI endpoints
│   ├── data/               # Loaders + validação
│   ├── fairness/           # Análise de viés
│   ├── features/           # Feature engineering
│   ├── interpretability/   # SHAP analysis
│   ├── models/             # Training + evaluation
│   └── tracking/           # MLflow integration
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_interpretability.ipynb
│   └── 04_fairness.ipynb
├── tests/
├── models/                 # Modelos salvos
├── reports/                # Métricas e figuras
├── pyproject.toml
└── Makefile
```

---

## Tecnologias

| Categoria | Ferramentas |
|-----------|-------------|
| **ML** | scikit-learn, XGBoost, LightGBM |
| **Otimização** | Optuna |
| **Interpretabilidade** | SHAP |
| **Fairness** | Fairlearn |
| **API** | FastAPI, Uvicorn |
| **Tracking** | MLflow |
| **Validação** | Pandera |
| **Visualização** | Matplotlib, Seaborn, Plotly |
| **Qualidade** | Black, Flake8, isort, mypy, pre-commit |
| **CI/CD** | GitHub Actions |
| **Ambiente** | Poetry |

---

## Comandos Makefile

| Comando | Descrição |
|---------|-----------|
| `make install` | Instalar dependências |
| `make train` | Treinar com Optuna |
| `make train-quick` | Treinar sem Optuna |
| `make api` | Iniciar API |
| `make mlflow` | Iniciar MLflow UI |
| `make fairness` | Executar análise de fairness |
| `make test` | Rodar testes |
| `make format` | Formatar código |
| `make lint` | Verificar linting |

---

## Referências

- [German Credit Dataset - UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Fairlearn Documentation](https://fairlearn.org/)

---

## Licença

Este projeto está sob a licença MIT.

---

## Autor

**Jorge** - Cientista de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/seu-perfil)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/seu-usuario)

