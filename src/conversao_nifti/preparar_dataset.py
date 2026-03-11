"""
Converte um dataset DICOM estruturado em um dataset NIfTI.

Estrutura esperada do dataset DICOM:
    DATASET/
        NOME_PACIENTE/
            PRÉ-OP/
                <UID_SÉRIE>/
                    <UID_SÉRIE_DICOM>/
                        *.dcm
            PÓS-OP/
                <UID_SÉRIE>/
                    <UID_SÉRIE_DICOM>/
                        *.dcm

O dataset NIfTI resultante terá a seguinte estrutura:
    nifti_dataset/
        NOME_PACIENTE/
            preop_ct.nii.gz
            posop_ct.nii.gz
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import os

SCRIPT_DIR = Path(__file__).parent.resolve()          # src/conversao_nifti/
SRC_DIR    = SCRIPT_DIR.parent                        # src/
PROJECT_ROOT = SRC_DIR.parent                         # modelo_ia/

sys.path.append(str(SCRIPT_DIR))   # permite: from ingestion import ...

from ingestion import convert_dicom_to_nifti

DICOM_PATH = str(PROJECT_ROOT / "data" / "dataset_dicom")
NIFTI_PATH = str(PROJECT_ROOT / "data" / "dataset_nifti")



def convert_dicom_series_to_nifti(dicom_dir: Path, output_nifti: Path, verbose: bool = True):
    """
    Converte uma série DICOM em um único NIfTI
    """
    convert_dicom_to_nifti(str(dicom_dir), str(output_nifti), verbose=verbose)


def find_series_dir(stage_dir: Path) -> Path:
    """
    Varre recursivamente todas as subpastas e retorna o diretório
    com o maior número de arquivos DICOM (série volumétrica principal).
    Aceita .dcm, .DCM e variantes como .mdt.DCM.jpg.
    """
    best_dir: Path | None = None
    best_count: int = 0

    def search_all(directory: Path) -> None:
        nonlocal best_dir, best_count
        try:
            items = list(directory.iterdir())
        except (PermissionError, OSError):
            return

        dicom_files = [f for f in items if f.is_file() and '.dcm' in f.name.lower()]
        if len(dicom_files) > best_count:
            best_count = len(dicom_files)
            best_dir = directory

        for subdir in items:
            if subdir.is_dir():
                search_all(subdir)

    search_all(stage_dir)

    if best_dir is None:
        raise RuntimeError(f"Nenhum arquivo DICOM encontrado em {stage_dir}")

    return best_dir


def build_nifti_dataset(dicom_root: Path, output_root: Path, pacientes: list[str] | None = None):
    all_patients = [p for p in dicom_root.iterdir() if p.is_dir()]

    if pacientes:
        pacientes_lower = [n.lower() for n in pacientes]
        patients = [p for p in all_patients if p.name.lower() in pacientes_lower]
        nao_encontrados = set(pacientes_lower) - {p.name.lower() for p in patients}
        if nao_encontrados:
            print(f"⚠️ Pacientes não encontrados no dataset: {', '.join(nao_encontrados)}")
    else:
        patients = all_patients

    for patient_dir in tqdm(patients, desc="Processando pacientes"):
        patient_name = patient_dir.name
        output_patient_dir = output_root / patient_name

        for stage_label, stage_name in [
            ("preop_ct.nii.gz", "PRÉ-OP"),
            ("posop_ct.nii.gz", "PÓS-OP")
        ]:
            stage_dir = patient_dir / stage_name

            if not stage_dir.exists():
                print(f"⚠️ {stage_name} não encontrado para {patient_name}")
                continue

            try:
                dicom_series_dir = find_series_dir(stage_dir)
                output_nifti = output_patient_dir / stage_label

                convert_dicom_series_to_nifti(
                    dicom_series_dir,
                    output_nifti
                )

            except Exception as e:
                print(f"Erro em {patient_name} ({stage_name}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte dataset DICOM em NIfTI."
    )
    parser.add_argument(
        "--pacientes",
        nargs="+",
        metavar="NOME",
        default=None,
        help="Nome(s) do(s) paciente(s) a converter. Se omitido, converte todos."
    )
    args = parser.parse_args()

    dicom_root = Path(DICOM_PATH)
    output_root = Path(NIFTI_PATH)

    build_nifti_dataset(dicom_root, output_root, pacientes=args.pacientes)
    print("\n✅ Conversão concluída com sucesso!")
