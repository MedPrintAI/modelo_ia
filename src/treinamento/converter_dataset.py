"""
Converte o dataset de pacientes (data/dataset_nifti/) para o formato nnU-Net.

Estrutura de entrada (por paciente):
    data/dataset_nifti/<Paciente>/
        ├── preop_ct.nii.gz      ← volume de TC
        └── lesao.nii.gz         ← máscara binária (0=normal, 1=lesão)

Estrutura de saída (formato nnU-Net):
    data/nnunet/Dataset001_BoneLesions/
        ├── dataset.json
        ├── imagesTr/
        │   └── bone_001_0000.nii.gz  (sufixo _0000 = canal 0 = CT)
        ├── labelsTr/
        │   └── bone_001.nii.gz
        └── imagesTs/
            └── bone_XXX_0000.nii.gz  (pacientes sem máscara)

Uso:
    python converter_dataset.py
    python converter_dataset.py --dataset-id 1 --split 0.8
"""

import argparse
import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()   # src/treinamento/
PROJECT_ROOT = SCRIPT_DIR.parent.parent           # modelo_ia/
DATA_DIR     = PROJECT_ROOT / "data"
NIFTI_DIR    = DATA_DIR / "dataset_nifti"


def converter_dataset(dataset_id: int = 1, split_treino: float = 0.8,
                      nome: str = "BoneLesions") -> Path:
    """
    Converte dataset_nifti/ → nnunet/Dataset{id:03d}_{nome}/.

    Args:
        dataset_id:    ID numérico do dataset (padrão: 1)
        split_treino:  Fração de pacientes para treino (restante vai para imagesTs)
        nome:          Nome do dataset nnU-Net
    Returns:
        Caminho para a pasta raiz do dataset nnU-Net gerado
    """
    nome_dataset = f"Dataset{dataset_id:03d}_{nome}"
    saida_dir    = DATA_DIR / "nnunet" / nome_dataset

    imagens_tr = saida_dir / "imagesTr"
    labels_tr  = saida_dir / "labelsTr"
    imagens_ts = saida_dir / "imagesTs"

    for d in [imagens_tr, labels_tr, imagens_ts]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"📁 Criando dataset: {nome_dataset}")
    print(f"   Origem : {NIFTI_DIR}")
    print(f"   Destino: {saida_dir}")

    # Coleta pacientes
    pacientes = sorted([p for p in NIFTI_DIR.iterdir() if p.is_dir()])
    print(f"\n👥 Pacientes encontrados: {len(pacientes)}")

    com_mascara = []
    sem_mascara = []
    
    for p in pacientes:
        if (p / "lesao.nii.gz").exists():
            com_mascara.append((p, p / "lesao.nii.gz"))
        elif (p / "lesao.nii").exists():
            com_mascara.append((p, p / "lesao.nii"))
        else:
            sem_mascara.append(p)

    print(f"   Com máscara (lesao.nii.gz ou .nii): {len(com_mascara)}")
    print(f"   Sem máscara (só CT):               {len(sem_mascara)}")

    if not com_mascara:
        print("\n⚠️  Nenhum paciente tem lesao.nii.gz!")
        print("   Anote a lesão no 3D Slicer e salve como lesao.nii.gz na pasta do paciente.")
        print("   Continuando com os CTs sem máscara em imagesTs...")

    # ── Pacientes COM máscara → imagesTr + labelsTr ──────────────────────
    n_treino = max(1, int(len(com_mascara) * split_treino))
    treino   = com_mascara[:n_treino]
    teste_cm = com_mascara[n_treino:]  # resto dos com_mascara vai para Ts

    copiados_tr = 0
    for idx, (pac, mask_path) in enumerate(treino, start=1):
        ct_path = pac / "preop_ct.nii.gz"
        if not ct_path.exists():
            print(f"   ⚠️  {pac.name}: preop_ct.nii.gz não encontrado, pulando.")
            continue

        prefixo = f"bone_{idx:03d}"
        dst_ct   = imagens_tr / f"{prefixo}_0000.nii.gz"
        dst_mask = labels_tr  / f"{prefixo}.nii.gz"

        shutil.copy2(ct_path, dst_ct)
        
        # Se for .nii (descomprimido), comprime ao copiar para .nii.gz
        if mask_path.suffix == '.nii':
            import gzip
            with open(mask_path, 'rb') as f_in, gzip.open(dst_mask, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(mask_path, dst_mask)
            
        print(f"   ✅ Treino  {idx:03d}: {pac.name}")
        copiados_tr += 1

    # ── Pacientes SEM máscara + resto com máscara → imagesTs ─────────────
    ts_pacientes = [(p, p/"preop_ct.nii.gz") for p in sem_mascara] + \
                   [(p, p/"preop_ct.nii.gz") for p, _ in teste_cm]

    copiados_ts = 0
    idx_ts = n_treino + 1

    # Pacientes sem máscara
    for pac in sem_mascara:
        ct_path = pac / "preop_ct.nii.gz"
        if not ct_path.exists():
            print(f"   ⚠️  {pac.name}: preop_ct.nii.gz não encontrado, pulando.")
            continue
        prefixo = f"bone_{idx_ts:03d}"
        dst_ct  = imagens_ts / f"{prefixo}_0000.nii.gz"
        shutil.copy2(ct_path, dst_ct)
        print(f"   📋 Teste   {idx_ts:03d}: {pac.name} (sem máscara)")
        copiados_ts += 1
        idx_ts += 1

    # Pacientes com máscara reservados para validação (fora do split de treino)
    for pac, _ in teste_cm:
        ct_path = pac / "preop_ct.nii.gz"
        if not ct_path.exists():
            print(f"   ⚠️  {pac.name}: preop_ct.nii.gz não encontrado, pulando.")
            continue
        prefixo = f"bone_{idx_ts:03d}"
        dst_ct  = imagens_ts / f"{prefixo}_0000.nii.gz"
        shutil.copy2(ct_path, dst_ct)
        print(f"   🔬 Teste   {idx_ts:03d}: {pac.name} (reservado p/ validação — tem máscara)")
        copiados_ts += 1
        idx_ts += 1

    # ── dataset.json ─────────────────────────────────────────────────────
    dataset_json = {
        "name":         nome,
        "description":  "Segmentação de lesões ósseas crânio-maxilofaciais",
        "tensorImageSize": "3D",
        "reference":    "",
        "licence":      "",
        "release":      "0.1.0",
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "lesao":      1
        },
        "numTraining":  copiados_tr,
        "file_ending":  ".nii.gz"
    }

    json_path = saida_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    # ── Resumo ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"✅ Conversão concluída!")
    print(f"   imagesTr : {copiados_tr} CT(s) com máscara")
    print(f"   imagesTs : {copiados_ts} CT(s) sem máscara")
    print(f"   dataset.json → {json_path}")
    print(f"\n{'=' * 50}")
    print("📌 Próximos passos:")
    print("   1. Comprima a pasta abaixo e suba para o Google Drive:")
    print(f"      {saida_dir}")
    print("   2. Abra o notebook treinar_colab.ipynb no Google Colab")
    print("   3. Ou, se tiver GPU local, rode:")
    print(f"      export nnUNet_raw={saida_dir.parent}")
    print(f"      nnUNetv2_plan_and_preprocess -d {dataset_id:03d} --verify_dataset_integrity")

    return saida_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte dataset de pacientes para o formato nnU-Net"
    )
    parser.add_argument("--dataset-id",  type=int,   default=1,
                        help="ID numérico do dataset (padrão: 1)")
    parser.add_argument("--split",       type=float, default=0.8,
                        help="Fração treino/total (padrão: 0.8)")
    parser.add_argument("--nome",        type=str,   default="BoneLesions",
                        help="Nome do dataset (padrão: BoneLesions)")
    args = parser.parse_args()

    converter_dataset(dataset_id=args.dataset_id,
                      split_treino=args.split,
                      nome=args.nome)
