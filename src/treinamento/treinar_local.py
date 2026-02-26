"""
Setup e treino local com GPU (alternativa ao Google Colab).

Execute este script uma vez para configurar o ambiente.
Depois use os comandos nnU-Net diretamente no terminal.

Uso:
    python treinar_local.py --setup     # instala dependências + configura env
    python treinar_local.py --treinar   # roda plan_and_preprocess + treino
    python treinar_local.py --avaliar   # avalia os resultados
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
SCRIPT_DIR          = Path(__file__).parent.resolve()   # src/treinamento/
PROJECT_ROOT        = SCRIPT_DIR.parent.parent           # modelo_ia/
DATA_DIR            = PROJECT_ROOT / "data"
NNUNET_RAW_DIR      = DATA_DIR / "nnunet"
NNUNET_PREPROC_DIR  = DATA_DIR / "nnunet_preprocessed"
RESULTS_DIR         = PROJECT_ROOT / "modelo_treinado"
DATASET_ID          = "001"


def definir_env() -> dict:
    """Retorna as variáveis de ambiente necessárias para o nnU-Net."""
    return {
        **os.environ,
        "nnUNet_raw":          str(NNUNET_RAW_DIR),
        "nnUNet_preprocessed": str(NNUNET_PREPROC_DIR),
        "nnUNet_results":      str(RESULTS_DIR),
    }


def run(cmd: list[str], env: dict | None = None) -> int:
    """Executa um comando e retorna o exit code."""
    print(f"\n▶  {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, env=env or definir_env())
    return result.returncode


def cmd_setup():
    """Instala PyTorch com CUDA e nnU-Net."""
    print("=" * 60)
    print("🛠️  Setup: instalando dependências")
    print("=" * 60)

    # Detecta versão CUDA disponível
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if "release 11" in nvcc.stdout:
            cuda_tag = "cu118"
        elif "release 12" in nvcc.stdout:
            cuda_tag = "cu121"
        else:
            cuda_tag = "cu121"  # padrão
    except FileNotFoundError:
        cuda_tag = "cu121"

    print(f"   CUDA detectado → usando torch com {cuda_tag}")

    torch_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    run([sys.executable, "-m", "pip", "install",
         "torch", "torchvision",
         "--index-url", torch_url])

    run([sys.executable, "-m", "pip", "install", "nnunetv2"])

    # Verificação
    code = run([sys.executable, "-c",
        "import torch; "
        "print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, '
               f'GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
    ])

    if code == 0:
        print("\n✅ Setup concluído!")
    else:
        print("\n❌ Erro no setup. Veja mensagens acima.")
    return code


def cmd_treinar(fold: int = 0, todos_folds: bool = False):
    """Roda plan_and_preprocess e treino nnU-Net."""
    print("=" * 60)
    print(f"🚀  Treinamento nnU-Net — Dataset {DATASET_ID}")
    print("=" * 60)

    # Verifica dataset
    dataset_dir = NNUNET_RAW_DIR / f"Dataset{DATASET_ID}_BoneLesions"
    if not dataset_dir.exists():
        print(f"❌ Dataset não encontrado: {dataset_dir}")
        print("   Execute primeiro: python converter_dataset.py")
        return 1

    images_tr = dataset_dir / "imagesTr"
    n_treino = len(list(images_tr.glob("*.nii.gz")))
    if n_treino == 0:
        print("❌ imagesTr/ está vazia! Nenhum caso de treino.")
        print("   Adicione máscaras (lesao.nii.gz) e rode converter_dataset.py")
        return 1
    print(f"   Casos de treino encontrados: {n_treino}")

    # Cria pastas
    NNUNET_PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = definir_env()

    # Passo 1: Plan and preprocess
    print("\n📐 Etapa 1/2: Planejamento e pré-processamento...")
    code = run(["nnUNetv2_plan_and_preprocess",
                "-d", DATASET_ID,
                "--verify_dataset_integrity",
                "-np", "4"], env=env)
    if code != 0:
        return code

    # Passo 2: Treino
    print("\n🧠 Etapa 2/2: Treinamento...")
    folds = list(range(5)) if todos_folds else [fold]
    for f in folds:
        print(f"\n   Treinando fold {f}...")
        code = run(["nnUNetv2_train",
                    DATASET_ID, "3d_fullres", str(f),
                    "--npz"], env=env)
        if code != 0:
            print(f"❌ Erro no fold {f}")
            return code

    print(f"\n✅ Treino concluído!")
    print(f"   Modelo salvo em: {RESULTS_DIR}")
    return 0


def cmd_avaliar():
    """Avalia o modelo e mostra o Dice Score."""
    print("=" * 60)
    print("📊  Avaliação do modelo")
    print("=" * 60)

    env = definir_env()
    return run(["nnUNetv2_find_best_configuration",
                DATASET_ID, "-c", "3d_fullres"], env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup e treino local nnU-Net com GPU"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup",    action="store_true",
                       help="Instala PyTorch CUDA + nnU-Net")
    group.add_argument("--treinar",  action="store_true",
                       help="Roda plan_and_preprocess + treino")
    group.add_argument("--avaliar",  action="store_true",
                       help="Avalia o modelo treinado (Dice Score)")

    parser.add_argument("--fold",       type=int, default=0,
                        help="Fold a treinar (0-4, padrão: 0)")
    parser.add_argument("--todos-folds", action="store_true",
                        help="Treinar os 5 folds sequencialmente")
    args = parser.parse_args()

    if args.setup:
        sys.exit(cmd_setup())
    elif args.treinar:
        sys.exit(cmd_treinar(fold=args.fold, todos_folds=args.todos_folds))
    elif args.avaliar:
        sys.exit(cmd_avaliar())
