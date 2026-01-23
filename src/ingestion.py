"""
Módulo de Ingestão - Conversão de DICOM para NIfTI

Este módulo lida com o carregamento de arquivos DICOM e conversão
para formato NIfTI, que é o padrão usado pela maioria dos modelos de IA médica.
"""

import os
from pathlib import Path
from typing import Optional
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from tqdm import tqdm


def convert_dicom_to_nifti(
    dicom_directory: str,
    output_path: str,
    verbose: bool = True
) -> str:
    """
    Converte uma série de arquivos DICOM para um único arquivo NIfTI.
    
    Args:
        dicom_directory: Caminho para o diretório contendo os arquivos DICOM
        output_path: Caminho de saída para o arquivo .nii.gz
        verbose: Se True, exibe progresso
        
    Returns:
        Caminho do arquivo NIfTI gerado
        
    Raises:
        FileNotFoundError: Se o diretório DICOM não existir
        ValueError: Se não houver arquivos DICOM válidos
    """
    dicom_path = Path(dicom_directory)
    if not dicom_path.exists():
        raise FileNotFoundError(f"Diretório DICOM não encontrado: {dicom_directory}")
    
    if verbose:
        print(f"Lendo arquivos DICOM de: {dicom_directory}")
    
    # Lê a série DICOM usando SimpleITK
    reader = sitk.ImageSeriesReader()
    
    # Obtém os IDs das séries DICOM no diretório
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_path))
    
    if not series_ids:
        raise ValueError(f"Nenhuma série DICOM encontrada em {dicom_directory}")
    
    if verbose:
        print(f"Encontradas {len(series_ids)} série(s) DICOM")
    
    # Usa a primeira série encontrada
    series_id = series_ids[0]
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path), series_id)
    
    if not dicom_names:
        raise ValueError(f"Nenhum arquivo DICOM válido encontrado na série {series_id}")
    
    if verbose:
        print(f"Processando {len(dicom_names)} arquivo(s) DICOM...")
    
    reader.SetFileNames(dicom_names)
    
    # Lê a imagem
    image = reader.Execute()
    
    if verbose:
        print(f"Dimensões da imagem: {image.GetSize()}")
        print(f"Espaçamento: {image.GetSpacing()}")
        print(f"Origem: {image.GetOrigin()}")
    
    # Converte para array numpy
    image_array = sitk.GetArrayFromImage(image)
    
    # Obtém metadados importantes
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    # Ajusta a ordem dos eixos (SimpleITK usa ZYX, NIfTI usa XYZ)
    # NIfTI espera (x, y, z) mas SimpleITK retorna (z, y, x)
    image_array = np.transpose(image_array, (2, 1, 0))
    
    # Cria a matriz de afim para NIfTI
    # A direção precisa ser ajustada também
    affine = np.eye(4)
    affine[:3, :3] = np.array(direction).reshape(3, 3)
    affine[:3, 3] = origin
    affine[0, 0] *= spacing[0]
    affine[1, 1] *= spacing[1]
    affine[2, 2] *= spacing[2]
    
    # Cria objeto NIfTI
    nifti_img = nib.Nifti1Image(image_array, affine)
    
    # Salva o arquivo
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Salvando NIfTI em: {output_path}")
    
    nib.save(nifti_img, output_path)
    
    if verbose:
        print("✓ Conversão concluída com sucesso!")
    
    return output_path


def validate_dicom_series(dicom_directory: str) -> dict:
    """
    Valida uma série DICOM e retorna informações sobre ela.
    
    Args:
        dicom_directory: Caminho para o diretório DICOM
        
    Returns:
        Dicionário com informações da série (dimensões, spacing, etc.)
    """
    dicom_path = Path(dicom_directory)
    if not dicom_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {dicom_directory}")
    
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_path))
    
    if not series_ids:
        return {"valid": False, "error": "Nenhuma série DICOM encontrada"}
    
    series_id = series_ids[0]
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path), series_id)
    
    if not dicom_names:
        return {"valid": False, "error": "Nenhum arquivo DICOM válido"}
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    return {
        "valid": True,
        "series_id": series_id,
        "num_files": len(dicom_names),
        "dimensions": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "pixel_type": image.GetPixelIDTypeAsString()
    }


