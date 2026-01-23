# MedPrintAI - IA para Reconstrução 3D Bucomaxilofacial

## 🎯 Objetivo

Automatizar o processo de reconstrução 3D bucomaxilofacial a partir de imagens médicas DICOM, utilizando técnicas de segmentação por IA e geração de modelos STL para impressão 3D.

## ⚙️ Fluxo de Trabalho

1. **Ingestão**: Conversão de dataset DICOM para dataset NIfTI

## 📋 Pré-requisitos

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
/
├── src/
│   ├── preparar_dataset.py            # Conversão de dataset DICOM para NIfTI
│   └── ingestion.py                   # Funções de ingestão de dados
├── data/
│   ├── dataset_dicom/                 # Coloque pastas DICOM aqui
│   └── dataset_nifti/                 # Dataset NIfTI gerado
├── .gitignore                         # Arquivo para ignorar arquivos
├── README.md                          # Este arquivo
└── requirements.txt                   # Dependências básicas
```

## 🚀 Como Executar

```bash
cd src
python preparar_dataset.py
```