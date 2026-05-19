# AM2 — Aprendizaje Máquina II

**Maestría en Inteligencia Artificial**  
Actividades del curso: Métodos de Kernel, Perceptrón Multicapa y Proyecto Aplicado.

> **Repositorio:** https://github.com/AlejoMoncada/am2-aprendizaje-maquina-ii

---

## Estructura del repositorio

```
AM2/
├── README.md
├── pyproject.toml              # Dependencias gestionadas con UV
├── data/
│   └── cfpb_features.parquet   # Features NLP (CFPB, 73k registros, 2.1 MB)
├── actividad1_metodos_kernel/
│   └── metodos_kernel_actividad.ipynb   # A1: SVM con kernels (RBF, poly, linear, sigmoid)
├── actividad2_perceptron/
│   └── perceptron_multicapa_actividad.ipynb  # A2: MLP con Keras (Dropout, L2, EarlyStopping)
├── proyecto1_heart_disease/
│   └── proyecto1_identificacion_problema.ipynb  # A3 alt: Heart Disease (UCI)
└── proyecto1_cfpb/
    └── proyecto1_cfpb.ipynb   # A3: CFPB Quejas Financieras (NLP + Kernel Methods)
```

## Requisitos

- **Python 3.13+**
- **UV** (gestor de paquetes): `pip install uv`

```bash
# Instalar dependencias
uv sync

# Ejecutar cualquier notebook
uv run jupyter notebook
```

## Actividades

### Actividad 1 — Métodos de Kernel
SVM con 4 kernels (linear, RBF, polinomial, sigmoide) sobre Breast Cancer Wisconsin. GridSearchCV para optimización de hiperparámetros.

### Actividad 2 — Perceptrón Multicapa
MLP con Keras/TensorFlow: comparación modelo base vs regularizado (Dropout + L2 + BatchNorm + EarlyStopping). Cuatro arquitecturas evaluadas.

### Actividad 3 — Proyecto Aplicado (CFPB)
Predicción de resolución de quejas financieras (Relief vs. Explanation) usando 18 features NLP generadas por pipeline propio. PCA + KDE demuestran no linealidad → justificación de kernel methods.

**Dataset:** `data/cfpb_features.parquet` (2.1 MB, 73,343 registros)  
**Origen:** Pipeline NLP de 6 etapas (spaCy, VADER, TextBlob, feature engineering)

## Tecnologías

- scikit-learn, TensorFlow/Keras, XGBoost
- pandas, numpy, scipy
- matplotlib, seaborn, altair
- pyarrow (Parquet)
- UV (gestión de dependencias)

## Ejecución

Todos los notebooks están **pre-ejecutados** con outputs visibles. Para re-ejecutar:

```bash
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb
```

---

**Autor:** William Alejandro Moncada Cifuentes  
**Fecha:** Mayo 2026
