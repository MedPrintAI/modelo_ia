"""
Inferência com modelo nnU-Net treinado.

Roda em CPU ou GPU. Colaboradores sem GPU podem usar este script
para gerar predições (será mais lento, mas funciona).

Uso:
    python inferencia.py --input caminho/ct.nii.gz --output caminho/mascara.nii.gz
    python inferencia.py --input pasta/com/cts/  --output pasta/saida/
    python inferencia.py --dataset               # processa dataset_nifti inteiro
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).parent.resolve()   # src/treinamento/
PROJECT_ROOT  = SCRIPT_DIR.parent.parent           # modelo_ia/
DATA_DIR      = PROJECT_ROOT / "data"
MODELO_DIR    = PROJECT_ROOT / "modelo_treinado"   # coloque aqui o modelo baixado do Colab
NNUNET_RAW    = DATA_DIR / "nnunet"
DATASET_ID    = "001"


def verificar_modelo() -> bool:
    """Verifica se o modelo treinado existe no caminho esperado."""
    if not MODELO_DIR.exists():
        print(f"❌ Modelo não encontrado em: {MODELO_DIR}")
        print()
        print("Para usar a inferência você precisa:")
        print("  1. Treinar o modelo no Google Colab (treinar_colab.ipynb)")
        print("  2. Baixar a pasta 'nnUNet_results' do Colab/Drive")
        print(f"  3. Colocar o conteúdo em: {MODELO_DIR}")
        return False
    return True


def inferir(input_path: str, output_path: str, usar_gpu: bool = False) -> int:
    """
    Roda a inferência nnU-Net em um arquivo ou pasta de CTs.

    Args:
        input_path:  Arquivo .nii.gz ou pasta com arquivos *_0000.nii.gz
        output_path: Arquivo ou pasta de saída
        usar_gpu:    True = usar GPU se disponível (False = CPU)
    Returns:
        Exit code do processo
    """
    if not verificar_modelo():
        return 1

    input_p  = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    # nnU-Net espera entrada como pasta
    if input_p.is_file():
        # Arquivo único: renomear para o padrão nnU-Net (_0000) em pasta temp
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        nome_sem_ext = input_p.name.replace(".nii.gz", "").replace(".nii", "")
        dst = tmp / f"{nome_sem_ext}_0000.nii.gz"
        shutil.copy2(input_p, dst)
        input_dir = str(tmp)
    else:
        input_dir = str(input_p)

    device = "gpu" if usar_gpu else "cpu"

    cmd = [
        sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data",
        "-i", input_dir,
        "-o", str(output_p),
        "-d", DATASET_ID,
        "-c", "3d_fullres",
        "-f", "all",       # usa todos os folds (melhor resultado)
        "--save_probabilities",
    ]

    env = {
        "nnUNet_raw":     str(NNUNET_RAW),
        "nnUNet_results": str(MODELO_DIR),
        "nnUNet_preprocessed": str(DATA_DIR / "nnunet_preprocessed"),
    }

    import os
    full_env = {**os.environ, **env}

    print(f"🤖 Rodando inferência nnU-Net...")
    print(f"   Entrada:  {input_dir}")
    print(f"   Saída:    {output_p}")
    print(f"   Device:   {device}")
    print()

    result = subprocess.run(cmd, env=full_env)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferência com modelo nnU-Net treinado"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="CT de entrada (.nii.gz) ou pasta com vários CTs")
    parser.add_argument("--output", "-o", default=None,
                        help="Caminho de saída (padrão: mesmo dir da entrada)")
    parser.add_argument("--gpu",    action="store_true",
                        help="Usar GPU se disponível (padrão: CPU)")
    args = parser.parse_args()

    input_p = Path(args.input)
    if args.output is None:
        if input_p.is_file():
            output_p = str(input_p.parent / "predicoes")
        else:
            output_p = str(input_p / "predicoes")
    else:
        output_p = args.output

    sys.exit(inferir(args.input, output_p, usar_gpu=args.gpu))
