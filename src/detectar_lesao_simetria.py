"""
Detecção Automática de Lesões Ósseas por Análise de Simetria Bilateral

Lógica:
    1. Carrega o volume NIfTI (tomografia pré-operatória)
    2. Segmenta apenas o osso (threshold de Hounsfield Units)
    3. Encontra o plano de simetria sagital mediano
    4. Espelha o lado saudável sobre o lado lesionado
    5. Calcula a diferença entre original e espelhado
    6. Regiões com diferença significativa = candidatos a lesão
    7. Salva a máscara de lesão como NIfTI

Uso:
    python detectar_lesao_simetria.py
    python detectar_lesao_simetria.py --input caminho/volume.nii.gz --output caminho/mascara.nii.gz
    python detectar_lesao_simetria.py --threshold-min 200 --diff-threshold 0.3
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def carregar_volume(nifti_path: str, verbose: bool = True) -> tuple:
    """
    Carrega um volume NIfTI e retorna o array de dados e o objeto de imagem.

    Returns:
        tupla (dados numpy, objeto nib.Nifti1Image)
    """
    path = Path(nifti_path)
    if not path.exists():
        # Tenta resolver relativo ao PROJECT_ROOT (modelo_ia/)
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        path_alt = project_root / nifti_path
        if path_alt.exists():
            path = path_alt
        else:
            raise FileNotFoundError(
                f"Arquivo não encontrado: {nifti_path}\n"
                f"   Também tentou: {path_alt}"
            )

    if verbose:
        print(f"📂 Carregando volume: {path.name}")

    img = nib.load(str(path))
    dados = img.get_fdata().astype(np.float32)

    if verbose:
        print(f"   Dimensões: {dados.shape}")
        print(f"   Espaçamento: {img.header.get_zooms()}")
        print(f"   Range de intensidade: [{dados.min():.0f}, {dados.max():.0f}] HU")

    return dados, img


def segmentar_osso(volume: np.ndarray, threshold_min: int = 200,
                   threshold_max: int = 3000, verbose: bool = True) -> np.ndarray:
    """
    Segmenta osso usando threshold de Hounsfield Units.

    Args:
        volume: Array 3D com valores de HU
        threshold_min: Valor mínimo de HU para osso (padrão: 200)
        threshold_max: Valor máximo de HU para osso (padrão: 3000)

    Returns:
        Máscara binária do osso (1 = osso, 0 = não-osso)
    """
    if verbose:
        print(f"\n🦴 Segmentando osso (HU: {threshold_min} - {threshold_max})...")

    mascara_osso = ((volume >= threshold_min) & (volume <= threshold_max)).astype(np.uint8)

    # Limpeza morfológica: remove pequenos ruídos
    struct = ndimage.generate_binary_structure(3, 1)
    mascara_osso = ndimage.binary_opening(mascara_osso, structure=struct, iterations=1).astype(np.uint8)

    # Mantém apenas o maior componente conectado (o crânio principal)
    labeled, num_features = ndimage.label(mascara_osso)
    if num_features > 1:
        tamanhos = ndimage.sum(mascara_osso, labeled, range(1, num_features + 1))
        maior_componente = np.argmax(tamanhos) + 1
        mascara_osso = (labeled == maior_componente).astype(np.uint8)

    total_voxels_osso = mascara_osso.sum()
    if verbose:
        print(f"   Voxels de osso encontrados: {total_voxels_osso:,}")
        porcentagem = (total_voxels_osso / volume.size) * 100
        print(f"   Porcentagem do volume: {porcentagem:.1f}%")

    return mascara_osso


def calcular_bounding_box(mascara: np.ndarray, padding: int = 20) -> tuple:
    """
    Calcula o bounding box de uma máscara 3D com padding.
    Retorna slices para cropping e as coordenadas do box.
    """
    coords = np.argwhere(mascara > 0)
    if len(coords) == 0:
        return None, mascara.shape

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Adiciona padding, respeitando limites do volume
    slices = tuple(
        slice(max(0, mn - padding), min(dim, mx + padding + 1))
        for mn, mx, dim in zip(mins, maxs, mascara.shape)
    )

    return slices, mascara.shape


def encontrar_plano_simetria(mascara_osso: np.ndarray, verbose: bool = True) -> int:
    """
    Encontra o plano sagital mediano (eixo X) da estrutura óssea.

    Usa o centro de massa da máscara de osso como referência.

    Args:
        mascara_osso: Máscara binária do osso

    Returns:
        Índice do plano sagital mediano no eixo X (eixo 0)
    """
    if verbose:
        print("\n📐 Encontrando plano de simetria sagital...")

    centro_massa = ndimage.center_of_mass(mascara_osso)
    plano_sagital = int(round(centro_massa[0]))

    if verbose:
        print(f"   Centro de massa: ({centro_massa[0]:.1f}, {centro_massa[1]:.1f}, {centro_massa[2]:.1f})")
        print(f"   Plano sagital mediano: X = {plano_sagital}")

    return plano_sagital


def espelhar_e_comparar(mascara_osso: np.ndarray, volume_original: np.ndarray,
                        plano_sagital: int, margem_superficie: int = 5,
                        verbose: bool = True) -> np.ndarray:
    """
    Espelha o volume pelo plano sagital e calcula a diferença.

    IMPORTANTE: Só considera "falta de osso" em voxels PRÓXIMOS à superfície
    óssea existente. Isso evita falsos positivos em regiões vazias distantes.

    Args:
        mascara_osso: Máscara binária do osso
        volume_original: Volume original com valores de HU
        plano_sagital: Índice do plano de simetria
        margem_superficie: Distância máxima (voxels) da superfície óssea
                          para considerar como candidato a lesão

    Returns:
        Mapa de diferença (float, 0 a 1) - quanto maior, mais provável lesão
    """
    if verbose:
        print("\n🔄 Espelhando e comparando lados...")

    dim_x = mascara_osso.shape[0]

    # Espelha o volume inteiro pelo eixo X (sagital)
    mascara_espelhada = np.flip(mascara_osso, axis=0)
    volume_espelhado = np.flip(volume_original, axis=0)

    # Ajuste: se o plano de simetria não está exatamente no centro,
    # precisamos compensar o deslocamento
    deslocamento = dim_x - 2 * plano_sagital
    if deslocamento != 0:
        if verbose:
            print(f"   Compensando deslocamento de {deslocamento} voxels...")
        mascara_espelhada = np.roll(mascara_espelhada, deslocamento, axis=0)
        volume_espelhado = np.roll(volume_espelhado, deslocamento, axis=0)

    # ---- REGIÃO DE INTERESSE: apenas perto da superfície do osso ----
    # Dilata a máscara de osso para criar uma "zona de busca" ao redor do osso.
    # Lesões reais (buracos, afundamentos) estão ADJACENTES ao osso existente.
    # Diferenças longe do osso são ruído de alinhamento.
    if verbose:
        print(f"   Criando zona de busca ({margem_superficie} voxels ao redor do osso)...")
    struct = ndimage.generate_binary_structure(3, 1)
    zona_busca = ndimage.binary_dilation(
        mascara_osso, structure=struct, iterations=margem_superficie
    ).astype(np.uint8)

    # Calcula diferença na máscara de osso (presença/ausência)
    diff_estrutural = np.zeros_like(mascara_osso, dtype=np.float32)

    # Caso 1: Osso presente no espelhado, ausente no original,
    # MAS apenas dentro da zona de busca (perto do osso existente)
    falta_osso = (mascara_espelhada == 1) & (mascara_osso == 0) & (zona_busca == 1)
    diff_estrutural[falta_osso] = 1.0

    # Caso 2: Diferença de densidade significativa (afundamento parcial)
    # Onde ambos têm osso, mas a densidade é muito diferente
    ambos_osso = (mascara_espelhada == 1) & (mascara_osso == 1)
    if ambos_osso.any():
        diff_densidade = np.abs(
            volume_original[ambos_osso].astype(np.float32) -
            volume_espelhado[ambos_osso].astype(np.float32)
        )
        # Usa percentil 95 para normalizar (evita outliers extremos)
        p95 = np.percentile(diff_densidade, 95)
        if p95 > 0:
            valores_norm = np.clip(diff_densidade / p95, 0, 1) * 0.5
            diff_estrutural[ambos_osso] = valores_norm

    if verbose:
        voxels_falta = falta_osso.sum()
        voxels_zona = zona_busca.sum() - mascara_osso.sum()
        voxels_diff_dens = (diff_estrutural[ambos_osso] > 0.2).sum() if ambos_osso.any() else 0
        print(f"   Zona de busca (excluindo osso): {voxels_zona:,} voxels")
        print(f"   Voxels com ausência de osso (na zona): {voxels_falta:,}")
        print(f"   Voxels com diferença de densidade: {voxels_diff_dens:,}")

    return diff_estrutural


def gerar_mascara_lesao(mapa_diferenca: np.ndarray, mascara_osso: np.ndarray,
                        diff_threshold: float = 0.5,
                        min_tamanho_lesao: int = 1000, verbose: bool = True) -> np.ndarray:
    """
    Gera a máscara final de lesão a partir do mapa de diferença.

    Args:
        mapa_diferenca: Mapa de diferença (0 a 1)
        mascara_osso: Máscara de osso original (para sanity check)
        diff_threshold: Limiar para considerar como lesão (padrão: 0.5)
        min_tamanho_lesao: Tamanho mínimo de uma lesão em voxels (padrão: 1000)

    Returns:
        Máscara binária de lesão (1 = lesão, 0 = saudável)
    """
    if verbose:
        print(f"\n🎯 Gerando máscara de lesão (threshold: {diff_threshold}, min tamanho: {min_tamanho_lesao})...")

    # Aplica threshold
    mascara = (mapa_diferenca >= diff_threshold).astype(np.uint8)

    # Suavização LEVE: fecha apenas pequenos buracos internos, SEM dilatar
    struct = ndimage.generate_binary_structure(3, 1)  # Conectividade mínima
    mascara = ndimage.binary_closing(mascara, structure=struct, iterations=1).astype(np.uint8)
    # NÃO faz dilation — era o principal culpado dos falsos positivos

    # Remove regiões pequenas (ruído / assimetria natural)
    labeled, num_features = ndimage.label(mascara)
    if num_features > 0:
        tamanhos = ndimage.sum(mascara, labeled, range(1, num_features + 1))
        for i, tamanho in enumerate(tamanhos, start=1):
            if tamanho < min_tamanho_lesao:
                mascara[labeled == i] = 0

    # ---- SANITY CHECK ----
    # Se a "lesão" é maior que 20% do osso total, provavelmente é falso positivo
    total_osso = mascara_osso.sum()
    total_lesao = mascara.sum()
    if total_osso > 0:
        ratio = total_lesao / total_osso
        if ratio > 0.20 and verbose:
            print(f"   ⚠️  AVISO: Lesão detectada = {ratio:.0%} do osso total!")
            print(f"   Isso pode indicar alinhamento impreciso ou assimetria natural.")
            print(f"   Considere aumentar --diff-threshold (atual: {diff_threshold})")

    # Recontagem final
    labeled_final, num_lesoes = ndimage.label(mascara)

    if verbose:
        print(f"   Regiões candidatas a lesão: {num_lesoes}")
        print(f"   Total de voxels de lesão: {total_lesao:,}")

        if num_lesoes > 0:
            # Ordena por tamanho (maior primeiro)
            tamanhos_finais = []
            for i in range(1, num_lesoes + 1):
                tamanho = (labeled_final == i).sum()
                tamanhos_finais.append((i, tamanho))
            tamanhos_finais.sort(key=lambda x: x[1], reverse=True)

            for idx, (label_id, tamanho) in enumerate(tamanhos_finais[:10]):  # Top 10
                print(f"   └─ Região {idx+1}: {tamanho:,} voxels")
            if num_lesoes > 10:
                print(f"   └─ ... e mais {num_lesoes - 10} regiões menores")

    return mascara


def salvar_mascara(mascara: np.ndarray, imagem_referencia: nib.Nifti1Image,
                   output_path: str, verbose: bool = True) -> str:
    """
    Salva a máscara de lesão como NIfTI, usando a geometria da imagem original.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    mascara_nifti = nib.Nifti1Image(mascara.astype(np.uint8), imagem_referencia.affine,
                                     imagem_referencia.header)
    nib.save(mascara_nifti, str(output))

    if verbose:
        print(f"\n💾 Máscara salva em: {output}")
        tamanho_mb = output.stat().st_size / (1024 * 1024)
        print(f"   Tamanho: {tamanho_mb:.2f} MB")

    return str(output)


def detectar_lesao(
    input_path: str,
    output_path: str | None = None,
    threshold_min: int = 200,
    threshold_max: int = 3000,
    diff_threshold: float = 0.5,
    min_tamanho_lesao: int = 1000,
    verbose: bool = True
) -> str:
    """
    Pipeline completo de detecção de lesão por simetria bilateral.

    Args:
        input_path: Caminho para o arquivo NIfTI de entrada
        output_path: Caminho para salvar a máscara (None = auto)
        threshold_min: HU mínimo para segmentação de osso
        threshold_max: HU máximo para segmentação de osso
        diff_threshold: Limiar de diferença para considerar lesão (0-1)
        min_tamanho_lesao: Tamanho mínimo de lesão em voxels
        verbose: Exibir progresso

    Returns:
        Caminho do arquivo de máscara gerado
    """
    if verbose:
        print("=" * 60)
        print("🔬 MedPrint AI - Detecção de Lesão por Simetria")
        print("=" * 60)

    # Auto-gerar caminho de saída
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.name.replace('.nii.gz', '').replace('.nii', '')}_lesao_mask.nii.gz")

    # 1. Carregar volume
    volume, img = carregar_volume(input_path, verbose=verbose)

    # 2. Segmentar osso
    mascara_osso = segmentar_osso(volume, threshold_min, threshold_max, verbose=verbose)

    if mascara_osso.sum() == 0:
        raise ValueError("Nenhuma estrutura óssea encontrada! Verifique o threshold de HU.")

    # 2.5. OTIMIZAÇÃO: Crop para bounding box do osso
    # Reduz volume de ~73M voxels para ~10-20M (3-5x mais rápido)
    bbox_slices, vol_shape = calcular_bounding_box(mascara_osso, padding=20)
    if bbox_slices is not None:
        vol_crop = volume[bbox_slices]
        osso_crop = mascara_osso[bbox_slices]
        if verbose:
            orig_size = np.prod(volume.shape)
            crop_size = np.prod(vol_crop.shape)
            print(f"\n✂️  Crop para bounding box do osso:")
            print(f"   Original: {volume.shape} = {orig_size:,} voxels")
            print(f"   Cropped:  {vol_crop.shape} = {crop_size:,} voxels ({crop_size/orig_size:.0%})")
    else:
        vol_crop = volume
        osso_crop = mascara_osso

    # 3. Encontrar plano de simetria (no volume cropped)
    plano = encontrar_plano_simetria(osso_crop, verbose=verbose)

    # 4. Espelhar e comparar (no volume cropped)
    mapa_diff = espelhar_e_comparar(osso_crop, vol_crop, plano, verbose=verbose)

    # 5. Gerar máscara de lesão (no volume cropped)
    mascara_lesao_crop = gerar_mascara_lesao(mapa_diff, osso_crop, diff_threshold, min_tamanho_lesao, verbose=verbose)

    # 5.5. Colocar resultado de volta no volume completo
    mascara_lesao = np.zeros(vol_shape, dtype=np.uint8)
    if bbox_slices is not None:
        mascara_lesao[bbox_slices] = mascara_lesao_crop
    else:
        mascara_lesao = mascara_lesao_crop

    # 6. Salvar
    output = salvar_mascara(mascara_lesao, img, output_path, verbose=verbose)

    if verbose:
        print()
        print("=" * 60)
        if mascara_lesao.sum() > 0:
            print("✅ Detecção concluída! Regiões candidatas a lesão encontradas.")
            print("⚠️  IMPORTANTE: Esta é uma PRÉ-ANOTAÇÃO automática.")
            print("   O especialista deve revisar e corrigir no 3D Slicer.")
        else:
            print("ℹ️  Nenhuma lesão significativa detectada pela análise de simetria.")
            print("   Isso pode significar:")
            print("   - O volume é simétrico (sem lesão lateral)")
            print("   - A lesão está na linha média")
            print("   - Os parâmetros precisam de ajuste")
        print("=" * 60)

    return output


def processar_dataset(dataset_dir: str, output_dir: str | None = None, **kwargs):
    """
    Processa todos os pacientes de um dataset NIfTI.

    Args:
        dataset_dir: Diretório raiz do dataset (contém pastas de pacientes)
        output_dir: Diretório de saída (None = salvar junto aos originais)
    """
    dataset_path = Path(dataset_dir)
    pacientes = [p for p in dataset_path.iterdir() if p.is_dir()]

    print(f"\n📋 Encontrados {len(pacientes)} paciente(s) no dataset\n")

    for paciente_dir in pacientes:
        preop = paciente_dir / "preop_ct.nii.gz"

        if not preop.exists():
            print(f"⚠️  {paciente_dir.name}: preop_ct.nii.gz não encontrado, pulando...")
            continue

        print(f"\n{'─' * 60}")
        print(f"👤 Paciente: {paciente_dir.name}")
        print(f"{'─' * 60}")

        if output_dir:
            out_dir = Path(output_dir) / paciente_dir.name
            out_path = str(out_dir / "preop_ct_lesao_mask.nii.gz")
        else:
            out_path = None  # Auto-gera junto ao original

        try:
            detectar_lesao(str(preop), output_path=out_path, **kwargs)
        except Exception as e:
            print(f"❌ Erro ao processar {paciente_dir.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detecção automática de lesões ósseas por análise de simetria bilateral"
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Arquivo NIfTI de entrada (ou usa o dataset padrão)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Arquivo NIfTI de saída para a máscara")
    parser.add_argument("--threshold-min", type=int, default=200,
                        help="HU mínimo para osso (padrão: 200)")
    parser.add_argument("--threshold-max", type=int, default=3000,
                        help="HU máximo para osso (padrão: 3000)")
    parser.add_argument("--diff-threshold", type=float, default=0.5,
                        help="Limiar de diferença para lesão, 0-1 (padrão: 0.5)")
    parser.add_argument("--min-tamanho", type=int, default=1000,
                        help="Tamanho mínimo de lesão em voxels (padrão: 1000)")
    parser.add_argument("--dataset", action="store_true",
                        help="Processar todo o dataset em vez de um arquivo individual")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATASET_NIFTI = PROJECT_ROOT / "data" / "dataset_nifti"

    kwargs = {
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "diff_threshold": args.diff_threshold,
        "min_tamanho_lesao": args.min_tamanho,
    }

    if args.dataset or args.input is None:
        # Processa o dataset inteiro
        processar_dataset(str(DATASET_NIFTI), **kwargs)
    else:
        # Processa arquivo individual
        detectar_lesao(args.input, output_path=args.output, **kwargs)
