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
import sys
import os

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.append(str(PROJECT_ROOT))

from ingestion import convert_dicom_to_nifti

DICOM_PATH = str(PROJECT_ROOT / "data" / "dataset_dicom")
NIFTI_PATH = str(PROJECT_ROOT / "data" / "dataset_nifti")


def convert_dicom_series_to_nifti(dicom_dir: Path, output_nifti: Path):
    """
    Converte uma série DICOM em um único NIfTI
    """
    convert_dicom_to_nifti(str(dicom_dir), str(output_nifti), verbose=False)


def find_series_dir(stage_dir: Path) -> Path:
    """
    Encontra recursivamente o diretório que contém os arquivos DICOM.
    Desce quantos níveis forem necessários até encontrar arquivos .dcm
    """
    def search_for_dicom(directory: Path) -> Path:
        """Procura recursivamente por arquivos DICOM"""
        try:
            items = list(directory.iterdir())
        except (PermissionError, OSError):
            raise RuntimeError(f"Não foi possível acessar {directory}")
        
        # Verifica se há arquivos DICOM neste diretório
        dicom_files = [f for f in items if f.is_file() and f.suffix.lower() == '.dcm']
        if dicom_files:
            return directory
        
        # Se não houver DCM, procura nos subdiretórios
        subdirs = [d for d in items if d.is_dir()]
        for subdir in subdirs:
            try:
                result = search_for_dicom(subdir)
                return result
            except RuntimeError:
                continue
        
        raise RuntimeError(f"Nenhum arquivo DICOM encontrado em {directory}")
    
    return search_for_dicom(stage_dir)


def build_nifti_dataset(dicom_root: Path, output_root: Path):
    patients = [p for p in dicom_root.iterdir() if p.is_dir()]

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
    dicom_root = Path(DICOM_PATH)
    output_root = Path(NIFTI_PATH)

    build_nifti_dataset(dicom_root, output_root)
    print("\n✅ Conversão concluída com sucesso!")
