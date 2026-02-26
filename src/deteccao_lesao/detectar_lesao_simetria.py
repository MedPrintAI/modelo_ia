"""
Detecção Automática de Lesões Ósseas por Análise de Simetria Bilateral (v2)

Melhorias em relação à v1:
  1. Plano de simetria otimizado por busca iterativa (em vez de simples centro de massa)
  2. Correção de inclinação da cabeça via PCA dos eixos inerciais do osso
  3. Comparação via casca óssea (cortical) em vez do volume inteiro
  4. Comparação por correlação cruzada normalizada (NCC) em patches locais
     em vez de diferença pixel-a-pixel (muito mais robusto a erros de alinhamento)
  5. Bounding box crop para acelerar processamento

Uso:
    python detectar_lesao_simetria.py --input volume.nii.gz
    python detectar_lesao_simetria.py --input volume.nii.gz --output mascara.nii.gz
    python detectar_lesao_simetria.py --dataset   # processa todos os pacientes
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Carregamento
# ---------------------------------------------------------------------------

def carregar_volume(nifti_path: str, verbose: bool = True) -> tuple:
    """Carrega um volume NIfTI. Retorna (array_float32, img_nibabel)."""
    path = Path(nifti_path)
    if not path.exists():
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent  # src/deteccao_lesao/ → src/ → modelo_ia/
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


# ---------------------------------------------------------------------------
# 2. Segmentação óssea
# ---------------------------------------------------------------------------

def segmentar_osso(volume: np.ndarray, threshold_min: int = 200,
                   threshold_max: int = 3000, verbose: bool = True) -> np.ndarray:
    """
    Segmenta osso via threshold de HU e mantém o maior componente conectado.
    """
    if verbose:
        print(f"\n🦴 Segmentando osso (HU: {threshold_min} – {threshold_max})...")

    mascara = ((volume >= threshold_min) & (volume <= threshold_max)).astype(np.uint8)

    # Remove ruído morfológico pequeno
    struct = ndimage.generate_binary_structure(3, 1)
    mascara = ndimage.binary_opening(mascara, structure=struct, iterations=1).astype(np.uint8)

    # Conserva apenas o maior componente (o crânio)
    labeled, n = ndimage.label(mascara)
    if n > 1:
        tamanhos = ndimage.sum(mascara, labeled, range(1, n + 1))
        mascara = (labeled == (np.argmax(tamanhos) + 1)).astype(np.uint8)

    if verbose:
        pct = mascara.sum() / volume.size * 100
        print(f"   Voxels de osso: {mascara.sum():,} ({pct:.1f}% do volume)")

    return mascara


# ---------------------------------------------------------------------------
# 3. Extração da casca óssea (cortical)
# ---------------------------------------------------------------------------

def extrair_casca_ossea(mascara_osso: np.ndarray, espessura: int = 4,
                        verbose: bool = True) -> np.ndarray:
    """
    Retorna apenas a superfície cortical do osso (borda externa).

    Técnica: diferença entre a máscara dilatada e a erodida, restrita ao osso.
    As lesões traumáticas (fraturas) estão na cortical, não no interior.

    Args:
        espessura: Espessura da casca em voxels (padrão: 4 ≈ 2 mm)
    """
    if verbose:
        print(f"\n🐚 Extraindo casca óssea (espessura {espessura} px)...")

    struct = ndimage.generate_binary_structure(3, 1)
    # Dilata levemente para incluir o tecido adjacente imediato
    dilatado = ndimage.binary_dilation(mascara_osso, structure=struct,
                                       iterations=espessura).astype(np.uint8)
    # Erode internamente para excluir o osso esponjoso profundo
    erodido = ndimage.binary_erosion(mascara_osso, structure=struct,
                                     iterations=espessura).astype(np.uint8)

    # Casca = zona entre erosão e dilatação, restrita ao próprio osso
    casca = (dilatado - erodido).clip(0, 1).astype(np.uint8)
    casca = (casca * mascara_osso).astype(np.uint8)  # garante que fica dentro do osso

    if verbose:
        print(f"   Voxels na casca: {casca.sum():,}")

    return casca


# ---------------------------------------------------------------------------
# 4. Bounding box
# ---------------------------------------------------------------------------

def calcular_bounding_box(mascara: np.ndarray, padding: int = 20) -> tuple:
    """Calcula slices de cropping para o bounding box da máscara, com padding."""
    coords = np.argwhere(mascara > 0)
    if len(coords) == 0:
        return None, mascara.shape

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    slices = tuple(
        slice(max(0, mn - padding), min(dim, mx + padding + 1))
        for mn, mx, dim in zip(mins, maxs, mascara.shape)
    )
    return slices, mascara.shape


# ---------------------------------------------------------------------------
# 5. Correção de inclinação via PCA
# ---------------------------------------------------------------------------

def corrigir_inclinacao(mascara_osso: np.ndarray, volume: np.ndarray,
                        verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta e corrige pequenas inclinações (roll) da cabeça usando PCA.

    O eixo principal de inércia do crânio deve apontar para cima (eixo Z).
    Qualquer desvio indica que o paciente estava inclinado no scanner.

    Returns:
        (mascara_corrigida, volume_corrigido) — rotacionados para ficar reto
    """
    if verbose:
        print("\n📐 Verificando inclinação da cabeça (PCA)...")

    # Coordenadas dos voxels de osso
    coords = np.argwhere(mascara_osso > 0).astype(np.float32)
    if len(coords) < 100:
        if verbose:
            print("   Poucos voxels — pulando correção de inclinação.")
        return mascara_osso, volume

    # Centraliza
    centro = coords.mean(axis=0)
    coords_c = coords - centro

    # Covariância e autovetores (eixos principais)
    cov = np.cov(coords_c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Ordena por eigenvalues decrescente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # O eixo 2 (Z) deveria ser o MENOR eixo de variação (crânio é achatado no Z)
    # O eixo 0 (X) deveria ser o eixo principal (sagital, esquerda-direita)
    eixo_principal = eigenvectors[:, 0]  # vetor com maior variância

    # Calcula ângulo de roll: entre eixo principal e o eixo X do scanner
    eixo_x = np.array([1.0, 0.0, 0.0])
    cos_angulo = np.dot(eixo_principal, eixo_x) / (
        np.linalg.norm(eixo_principal) * np.linalg.norm(eixo_x)
    )
    angulo_graus = np.degrees(np.arccos(np.clip(abs(cos_angulo), 0, 1)))

    if verbose:
        print(f"   Ângulo de inclinação detectado: {angulo_graus:.1f}°")

    # Só corrige se inclinação significativa (> 2°) e não excessiva (< 20°)
    if 2.0 < angulo_graus < 20.0:
        if verbose:
            print(f"   ✅ Aplicando correção de {angulo_graus:.1f}°...")

        # Calcula ângulo de rotação no plano XY (roll)
        angulo_rad = np.arctan2(eixo_principal[1], eixo_principal[0])

        # scipy.ndimage.rotate opera num plano 2D; aplica em cada fatia Z
        angulo_deg = np.degrees(angulo_rad)
        mascara_corrigida = ndimage.rotate(
            mascara_osso, -angulo_deg, axes=(0, 1), reshape=False,
            order=0, mode='constant', cval=0
        ).astype(np.uint8)
        volume_corrigido = ndimage.rotate(
            volume, -angulo_deg, axes=(0, 1), reshape=False,
            order=1, mode='constant', cval=volume.min()
        )
        return mascara_corrigida, volume_corrigido
    else:
        if verbose:
            print("   Inclinação dentro do tolerável — sem correção necessária.")
        return mascara_osso, volume


# ---------------------------------------------------------------------------
# 6. Plano de simetria otimizado
# ---------------------------------------------------------------------------

def encontrar_plano_simetria_otimizado(mascara_osso: np.ndarray,
                                       janela_busca: int = 15,
                                       verbose: bool = True) -> int:
    """
    Encontra o plano sagital que MAXIMIZA a simetria bilateral.

    Estratégia:
      - Parte do centro de massa como estimativa inicial.
      - Testa ±janela_busca posições de plano no eixo X.
      - Para cada candidato, calcula a "pontuação de simetria":
        correlação entre a projeção sagital do lado esquerdo e do direito.
      - Retorna o plano com maior correlação.

    Args:
        janela_busca: quantos voxels testar em torno do CoM (padrão: ±15)
    """
    if verbose:
        print("\n📐 Buscando plano de simetria otimizado...")

    dim_x = mascara_osso.shape[0]
    centro_massa = ndimage.center_of_mass(mascara_osso)
    com_x = int(round(centro_massa[0]))

    if verbose:
        print(f"   Centro de massa: X = {com_x}, busca: ±{janela_busca} px")

    # Projeta a máscara no plano YZ (soma ao longo do X)
    # para comparação rápida de densidade entre lados
    melhor_plano = com_x
    melhor_score = -1.0

    for delta in range(-janela_busca, janela_busca + 1):
        plano_candidato = com_x + delta
        if plano_candidato <= 0 or plano_candidato >= dim_x - 1:
            continue

        # Divide em esquerda e direita
        lado_esq = mascara_osso[:plano_candidato, :, :]
        lado_dir = mascara_osso[plano_candidato:, :, :]

        # Equaliza o tamanho (pega o menor dos dois lados)
        tamanho_min = min(lado_esq.shape[0], lado_dir.shape[0])
        if tamanho_min < 5:
            continue

        lado_esq_eq = lado_esq[-tamanho_min:, :, :]
        lado_dir_eq = lado_dir[:tamanho_min, :, :]
        lado_dir_flip = np.flip(lado_dir_eq, axis=0)

        # Calcula correlação das projeções YZ (soma por X → plano YZ)
        proj_esq = lado_esq_eq.sum(axis=0).astype(np.float32)
        proj_dir = lado_dir_flip.sum(axis=0).astype(np.float32)

        # Normaliza para média 0
        proj_esq -= proj_esq.mean()
        proj_dir -= proj_dir.mean()

        std_esq = proj_esq.std()
        std_dir = proj_dir.std()
        if std_esq < 1e-6 or std_dir < 1e-6:
            continue

        # Correlação de Pearson normalizada
        score = float(np.sum(proj_esq * proj_dir) / (proj_esq.size * std_esq * std_dir))

        if score > melhor_score:
            melhor_score = score
            melhor_plano = plano_candidato

    if verbose:
        print(f"   ✅ Melhor plano: X = {melhor_plano}  (score={melhor_score:.4f})")

    return melhor_plano


# ---------------------------------------------------------------------------
# 7. Comparação via NCC local em patches
# ---------------------------------------------------------------------------

def comparar_por_ncc_local(mascara_osso: np.ndarray, volume_original: np.ndarray,
                           plano_sagital: int, tamanho_patch: int = 16,
                           verbose: bool = True) -> np.ndarray:
    """
    Calcula um mapa de assimetria usando correlação cruzada normalizada (NCC)
    em patches 3D locais.

    Vantagem sobre diferença pixel-a-pixel: um patch detecta padrão local,
    sendo tolerante a deslocamentos de ±2-3 px (ruído de alinhamento normal).

    Estratégia:
      - Para cada patch em grade 3D, calcula NCC entre o patch original e o
        patch correspondente no volume espelhado.
      - NCC ≈ 1.0 → patch altamente simétrico (região saudável)
      - NCC ≈ 0.0 → patch assimétrico (candidato a lesão)
      - Mapa de assimetria = 1 - NCC, interpolado para resolução original

    Args:
        tamanho_patch: Lado do cubo de patch (padrão: 16 px)
    """
    if verbose:
        print(f"\n🔬 Comparando por NCC local (patches {tamanho_patch}³)...")

    dim_x = mascara_osso.shape[0]
    shape = mascara_osso.shape

    # Espelha pelo plano sagital
    deslocamento = dim_x - 2 * plano_sagital
    vol_espelhado = np.flip(volume_original, axis=0)
    mascara_espelhada = np.flip(mascara_osso, axis=0)

    if deslocamento != 0:
        vol_espelhado = np.roll(vol_espelhado, deslocamento, axis=0)
        mascara_espelhada = np.roll(mascara_espelhada, deslocamento, axis=0)

    # Grade de centros dos patches (passo = metade do patch → 50% sobreposição)
    passo = tamanho_patch // 2
    centros_x = np.arange(passo, shape[0] - passo, passo)
    centros_y = np.arange(passo, shape[1] - passo, passo)
    centros_z = np.arange(passo, shape[2] - passo, passo)

    # Volume de assimetria (resolução reduzida → depois interpola)
    mapa_low = np.zeros((len(centros_x), len(centros_y), len(centros_z)),
                        dtype=np.float32)

    meio = tamanho_patch // 2

    for ix, cx in enumerate(centros_x):
        for iy, cy in enumerate(centros_y):
            for iz, cz in enumerate(centros_z):
                # Extrai patch original
                slc = (
                    slice(cx - meio, cx + meio),
                    slice(cy - meio, cy + meio),
                    slice(cz - meio, cz + meio),
                )
                patch_orig = volume_original[slc].astype(np.float32).ravel()
                patch_esph = vol_espelhado[slc].astype(np.float32).ravel()
                mask_orig = mascara_osso[slc].ravel()
                mask_esph = mascara_espelhada[slc].ravel()

                # Só avalia patches com osso em pelo menos um dos lados
                tem_osso = (mask_orig.sum() + mask_esph.sum()) > (tamanho_patch ** 3 * 0.05)
                if not tem_osso:
                    mapa_low[ix, iy, iz] = 0.0
                    continue

                # NCC normalizada
                m1 = patch_orig - patch_orig.mean()
                m2 = patch_esph - patch_esph.mean()
                denom = np.sqrt((m1 ** 2).sum() * (m2 ** 2).sum())
                if denom < 1e-6:
                    mapa_low[ix, iy, iz] = 0.0
                else:
                    ncc = float(np.dot(m1, m2) / denom)
                    ncc = np.clip(ncc, -1.0, 1.0)
                    # Assimetria = 1 - NCC (quanto menos correlacionado, mais assimétrico)
                    assimetria = (1.0 - ncc) / 2.0  # normaliza para [0, 1]
                    mapa_low[ix, iy, iz] = assimetria

    # Interpola de volta para a resolução original
    fatores_zoom = (
        shape[0] / mapa_low.shape[0],
        shape[1] / mapa_low.shape[1],
        shape[2] / mapa_low.shape[2],
    )
    mapa_diff = ndimage.zoom(mapa_low, fatores_zoom, order=1)
    # Recorta para o shape exato caso o zoom arredonde diferente
    mapa_diff = mapa_diff[:shape[0], :shape[1], :shape[2]]
    # Garante que regiões sem osso fiquem em 0
    zona_osso = ndimage.binary_dilation(mascara_osso, iterations=4).astype(bool)
    mapa_diff[~zona_osso] = 0.0

    if verbose:
        print(f"   Patches avaliados: {len(centros_x) * len(centros_y) * len(centros_z):,}")
        print(f"   Assimetria máxima: {mapa_diff.max():.3f}, média: {mapa_diff.mean():.4f}")

    return mapa_diff.astype(np.float32)


# ---------------------------------------------------------------------------
# 8. Geração da máscara final
# ---------------------------------------------------------------------------

def gerar_mascara_lesao(mapa_diferenca: np.ndarray, mascara_osso: np.ndarray,
                        casca_ossea: np.ndarray,
                        diff_threshold: float = 0.35,
                        min_tamanho_lesao: int = 500,
                        verbose: bool = True) -> np.ndarray:
    """
    Gera a máscara binária de lesão a partir do mapa de assimetria.

    Restringe a detecção à casca óssea (cortical) para evitar falsos positivos
    em tecidos moles internos.
    """
    if verbose:
        print(f"\n🎯 Gerando máscara (threshold={diff_threshold}, min={min_tamanho_lesao} vx)...")

    # Aplica threshold APENAS na zona da casca óssea mais uma margem pequena
    zona_valida = ndimage.binary_dilation(casca_ossea, iterations=3).astype(bool)
    mascara = np.zeros_like(mapa_diferenca, dtype=np.uint8)
    mascara[zona_valida & (mapa_diferenca >= diff_threshold)] = 1

    # Fechamento leve para unir fragmentos próximos
    struct = ndimage.generate_binary_structure(3, 1)
    mascara = ndimage.binary_closing(mascara, structure=struct, iterations=2).astype(np.uint8)

    # Remove regiões pequenas (ruído / assimetria natural leve)
    labeled, n_feat = ndimage.label(mascara)
    if n_feat > 0:
        tamanhos = ndimage.sum(mascara, labeled, range(1, n_feat + 1))
        for i, tam in enumerate(tamanhos, start=1):
            if tam < min_tamanho_lesao:
                mascara[labeled == i] = 0

    # Sanity check
    total_osso = mascara_osso.sum()
    total_lesao = mascara.sum()
    ratio = total_lesao / total_osso if total_osso > 0 else 0

    labeled_final, n_lesoes = ndimage.label(mascara)

    if verbose:
        print(f"   Regiões de lesão: {n_lesoes}")
        print(f"   Voxels de lesão: {total_lesao:,} ({ratio:.1%} do osso)")
        if ratio > 0.15:
            print(f"   ⚠️  Ratio alto ({ratio:.0%})! Considere --diff-threshold maior.")
        if n_lesoes > 0:
            tamanhos_finais = sorted(
                [(i, int((labeled_final == i).sum())) for i in range(1, n_lesoes + 1)],
                key=lambda x: x[1], reverse=True
            )
            for idx, (_, tam) in enumerate(tamanhos_finais[:8]):
                print(f"   └─ Região {idx+1}: {tam:,} voxels")
            if n_lesoes > 8:
                print(f"   └─ ... +{n_lesoes - 8} regiões menores")

    return mascara


def filtrar_bilateral(mascara: np.ndarray, plano_sagital: int,
                      sobreposicao_min: float = 0.15,
                      margem_flip: int = 3,
                      verbose: bool = True) -> np.ndarray:
    """
    Remove regiões que aparecem em AMBOS os lados do plano sagital.

    Justificativa clínica:
    - Lesões traumáticas são UNILATERAIS (fratura em um lado só).
    - Estruturas naturalmente assimétricas (seios paranasais, mastoides)
      tendem a gerar detecções BILATERAIS similares.
    - Portanto: se uma região tem um "espelho" correspondente no lado oposto,
      ela NÃO é uma lesão real — é assimetria anatômica normal.

    Args:
        sobreposicao_min: Fração mínima de sobreposição com o espelho
                          para considerar como "par bilateral" (padrão: 15%)
        margem_flip:      Margem de tolerância no deslocamento do espelho (px)
    """
    if verbose:
        print("\n🪞 Aplicando filtro bilateral (remove regiões simétricas)...")

    dim_x = mascara.shape[0]
    deslocamento = dim_x - 2 * plano_sagital

    # Espelha a máscara de lesão pelo plano sagital
    mascara_flip = np.flip(mascara, axis=0)
    if deslocamento != 0:
        mascara_flip = np.roll(mascara_flip, deslocamento, axis=0)

    # Dilata levemente o espelho para compensar imprecisão de alinhamento
    struct = ndimage.generate_binary_structure(3, 1)
    mascara_flip_dilatada = ndimage.binary_dilation(
        mascara_flip, structure=struct, iterations=margem_flip
    ).astype(np.uint8)

    # Identifica regiões e verifica cada uma contra o espelho
    labeled, n = ndimage.label(mascara)
    mascara_filtrada = mascara.copy()
    removidas = 0

    for i in range(1, n + 1):
        regiao = (labeled == i)
        tam = regiao.sum()
        # Voxels da região que têm sobreposição com o espelho dilatado
        sobreposicao = (regiao & (mascara_flip_dilatada > 0)).sum()
        ratio_sob = sobreposicao / tam if tam > 0 else 0

        if ratio_sob >= sobreposicao_min:
            # Tem correspondência bilateral → provavelmente é assimetria normal
            mascara_filtrada[labeled == i] = 0
            removidas += 1

    labeled_final, n_final = ndimage.label(mascara_filtrada)

    if verbose:
        print(f"   Regiões antes do filtro: {n}")
        print(f"   Regiões removidas (bilaterais): {removidas}")
        print(f"   Regiões restantes (unilaterais = candidatos a lesão): {n_final}")
        if n_final > 0:
            for i in range(1, n_final + 1):
                tam = int((labeled_final == i).sum())
                lado = "ESQUERDA" if ndimage.center_of_mass(labeled_final == i)[0] < plano_sagital else "DIREITA"
                print(f"   └─ Região {i}: {tam:,} voxels — lado {lado}")

    return mascara_filtrada




# ---------------------------------------------------------------------------
# 9. Salvar
# ---------------------------------------------------------------------------

def salvar_mascara(mascara: np.ndarray, img_ref: nib.Nifti1Image,
                   output_path: str, verbose: bool = True) -> str:
    """Salva a máscara como NIfTI com a geometria do volume original."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nifti_out = nib.Nifti1Image(mascara.astype(np.uint8), img_ref.affine, img_ref.header)
    nib.save(nifti_out, str(output))
    if verbose:
        mb = output.stat().st_size / (1024 * 1024)
        print(f"\n\U0001f4be Máscara salva: {output}  ({mb:.1f} MB)")
    return str(output)


def salvar_mapa_assimetria(mapa: np.ndarray, img_ref: nib.Nifti1Image,
                           output_path: str, verbose: bool = True) -> str:
    """
    Salva o mapa bruto de assimetria NCC como NIfTI float32.
    Permite inspecionar visualmente no 3D Slicer com uma escala de cor.
    Valores próximos de 0 = simétrico (saudável), próximos de 1 = assimétrico.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nifti_out = nib.Nifti1Image(mapa.astype(np.float32), img_ref.affine, img_ref.header)
    nib.save(nifti_out, str(output))
    if verbose:
        mb = output.stat().st_size / (1024 * 1024)
        print(f"\U0001f4ca Mapa de assimetria salvo: {output}  ({mb:.1f} MB)")
    return str(output)


def calcular_threshold_automatico(mapa_diff: np.ndarray, casca_ossea: np.ndarray,
                                  percentil: float = 97.0, verbose: bool = True) -> float:
    """
    Calcula automaticamente o limiar de detecção por percentil.

    Estratégia: usa o percentil P dos valores de assimetria NA CASCA óssea.
    Voxels acima do P-ésimo percentil = outliers = candidatos a lesão.

    Distribuições de assimetria normal são muito assimétricas à direita
    (grande maioria perto de 0, poucas regiões muito assimétricas),
    portanto percentil é muito mais robusto que média+sigma.

    Args:
        percentil: percentil de corte (padrão: 97 → top 3% de assimetria)
    Returns:
        threshold calculado, limitado entre 0.40 e 0.80
    """
    valores_casca = mapa_diff[casca_ossea > 0]
    valores_casca = valores_casca[valores_casca > 0.01]  # exclui zeros exatos

    if len(valores_casca) < 100:
        if verbose:
            print("   Poucos valores na casca, usando threshold padrão: 0.60")
        return 0.60

    # Percentis para entender a distribuição
    p50  = float(np.percentile(valores_casca, 50))
    p75  = float(np.percentile(valores_casca, 75))
    p90  = float(np.percentile(valores_casca, 90))
    p95  = float(np.percentile(valores_casca, 95))
    p97  = float(np.percentile(valores_casca, 97))
    p99  = float(np.percentile(valores_casca, 99))
    pmax = float(valores_casca.max())

    threshold = float(np.clip(np.percentile(valores_casca, percentil), 0.40, 0.80))

    if verbose:
        print(f"   Distribuição de assimetria na casca:")
        print(f"   p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}  "
              f"p95={p95:.3f}  p97={p97:.3f}  p99={p99:.3f}  max={pmax:.3f}")
        print(f"   → Threshold automático (p{percentil:.0f}): {threshold:.4f}")
        print(f"   (use --diff-threshold para ajustar manualmente)")

    return threshold


# ---------------------------------------------------------------------------
# 10. Pipeline principal
# ---------------------------------------------------------------------------

def detectar_lesao(
    input_path: str,
    output_path: str | None = None,
    threshold_min: int = 200,
    threshold_max: int = 3000,
    diff_threshold: float | None = None,   # None = auto
    n_sigma: float = 3.0,                  # para auto-threshold
    min_tamanho_lesao: int = 500,
    espessura_casca: int = 4,
    tamanho_patch: int = 16,
    corrigir_tilt: bool = True,
    salvar_mapa: bool = False,
    verbose: bool = True,
) -> str:
    """
    Pipeline de detecção de lesões ósseas por simetria bilateral (v2).

    Args:
        input_path:        Arquivo NIfTI de entrada
        output_path:       Saída (None = automático)
        threshold_min:     HU mínimo para osso (padrão: 200)
        threshold_max:     HU máximo para osso (padrão: 3000)
        diff_threshold:    Limiar de assimetria NCC 0–1 (None = auto)
        n_sigma:           Para auto-threshold: média + N*sigma (padrão: 3.0)
        min_tamanho_lesao: Tamanho mínimo de lesão em voxels (padrão: 500)
        espessura_casca:   Espessura da casca óssea em px (padrão: 4)
        tamanho_patch:     Lado do cubo de patch NCC em px (padrão: 16)
        corrigir_tilt:     Corrigir inclinação da cabeça via PCA (padrão: True)
        salvar_mapa:       Salvar mapa bruto de assimetria NCC (padrão: False)
    """
    if verbose:
        print("=" * 60)
        print("🔬  MedPrint AI — Detecção de Lesão por Simetria v2")
        print("=" * 60)

    # Resolve o caminho real do arquivo (pode ser relativo → absoluto)
    # Isso garante que a saída vá junto ao CT original,
    # independente de onde o script é executado.
    input_resolved = Path(input_path).resolve()
    if not input_resolved.exists():
        # Tenta relativo ao project root (modelo_ia/)
        script_dir   = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        alt = project_root / input_path
        if alt.exists():
            input_resolved = alt.resolve()

    # Auto-gerar saída junto ao CT original
    if output_path is None:
        stem = input_resolved.name.replace('.nii.gz', '').replace('.nii', '')
        output_path = str(input_resolved.parent / f"{stem}_lesao_mask.nii.gz")

    # ── 1. Carregar ──────────────────────────────────────────────────────
    volume, img = carregar_volume(input_path, verbose=verbose)

    # ── 2. Segmentar osso ─────────────────────────────────────────────────
    mascara_osso = segmentar_osso(volume, threshold_min, threshold_max, verbose=verbose)
    if mascara_osso.sum() == 0:
        raise ValueError("Nenhum osso encontrado. Verifique os thresholds de HU.")

    # ── 3. Bounding box (acelera tudo) ────────────────────────────────────
    bbox_slices, vol_shape = calcular_bounding_box(mascara_osso, padding=20)
    if bbox_slices is not None:
        vol_c = volume[bbox_slices]
        osso_c = mascara_osso[bbox_slices]
        if verbose:
            orig = np.prod(volume.shape)
            crop = np.prod(vol_c.shape)
            print(f"\n✂️  Crop: {volume.shape} → {vol_c.shape} ({crop/orig:.0%} do volume)")
    else:
        vol_c, osso_c = volume, mascara_osso

    # ── 4. Correção de inclinação ─────────────────────────────────────────
    if corrigir_tilt:
        osso_c, vol_c = corrigir_inclinacao(osso_c, vol_c, verbose=verbose)

    # ── 5. Extrai casca óssea ─────────────────────────────────────────────
    casca_c = extrair_casca_ossea(osso_c, espessura=espessura_casca, verbose=verbose)

    # ── 6. Encontra melhor plano de simetria ──────────────────────────────
    plano = encontrar_plano_simetria_otimizado(osso_c, verbose=verbose)

    # ── 7. Comparação NCC local ───────────────────────────────────────────
    mapa_diff = comparar_por_ncc_local(osso_c, vol_c, plano,
                                       tamanho_patch=tamanho_patch, verbose=verbose)

    # ── 7.5 Auto-threshold (se não especificado manualmente) ─────────────
    if diff_threshold is None:
        if verbose:
            print("\n\U0001f4ca Calculando threshold automático (p97 da casca óssea)...")
        diff_threshold = calcular_threshold_automatico(
            mapa_diff, casca_c, verbose=verbose
        )
        if verbose:
            print(f"   Threshold escolhido: {diff_threshold:.4f}")

    # ── 7.6 Salvar mapa bruto (opcional) ─────────────────────────────────
    if salvar_mapa:
        mapa_completo = np.zeros(vol_shape, dtype=np.float32)
        if bbox_slices is not None:
            mapa_completo[bbox_slices] = mapa_diff
        else:
            mapa_completo = mapa_diff
        stem_mapa = input_resolved.name.replace('.nii.gz', '').replace('.nii', '')
        mapa_path = str(input_resolved.parent / f"{stem_mapa}_assimetria.nii.gz")
        salvar_mapa_assimetria(mapa_completo, img, mapa_path, verbose=verbose)

    # ── 8. Gera máscara ───────────────────────────────────────────────────
    mascara_lesao_c = gerar_mascara_lesao(
        mapa_diff, osso_c, casca_c,
        diff_threshold=diff_threshold,
        min_tamanho_lesao=min_tamanho_lesao,
        verbose=verbose,
    )

    # ── 8.5. Filtro bilateral: remove regiões simétricas (falsos positivos) ─
    # Lesões reais são UNILATERAIS. Qualquer detecção que apareça bilateral-
    # mente (ambos os lados do plano sagital) é assimetria anatômica normal.
    mascara_lesao_c = filtrar_bilateral(mascara_lesao_c, plano, verbose=verbose)


    # ── 9. Reconstrói no volume completo ──────────────────────────────────
    mascara_lesao = np.zeros(vol_shape, dtype=np.uint8)
    if bbox_slices is not None:
        mascara_lesao[bbox_slices] = mascara_lesao_c
    else:
        mascara_lesao = mascara_lesao_c

    # ── 10. Salva ─────────────────────────────────────────────────────────
    output = salvar_mascara(mascara_lesao, img, output_path, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        if mascara_lesao.sum() > 0:
            print("✅ Pré-anotação gerada com sucesso!")
            print("   Abra no 3D Slicer para revisar e corrigir.")
        else:
            print("ℹ️  Nenhuma lesão detectada.")
            print("   Tente diminuir --diff-threshold (atual: {diff_threshold})")
        print("=" * 60)

    return output


# ---------------------------------------------------------------------------
# 11. Processar dataset completo
# ---------------------------------------------------------------------------

def processar_dataset(dataset_dir: str, output_dir: str | None = None, **kwargs):
    """Processa todos os pacientes do dataset."""
    dataset_path = Path(dataset_dir)
    pacientes = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
    print(f"\n📋 Encontrados {len(pacientes)} paciente(s)\n")

    for pac in pacientes:
        preop = pac / "preop_ct.nii.gz"
        if not preop.exists():
            print(f"⚠️  {pac.name}: preop_ct.nii.gz não encontrado, pulando...")
            continue

        print(f"\n{'─' * 60}")
        print(f"👤 Paciente: {pac.name}")
        print(f"{'─' * 60}")

        out_path = None
        if output_dir:
            out_dir = Path(output_dir) / pac.name
            out_path = str(out_dir / "preop_ct_lesao_mask.nii.gz")

        try:
            detectar_lesao(str(preop), output_path=out_path, **kwargs)
        except Exception as e:
            print(f"❌ Erro: {e}")


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detecção de lesões ósseas por análise de simetria bilateral (v2)"
    )
    parser.add_argument("--input", "-i", default=None,
                        help="Arquivo NIfTI de entrada")
    parser.add_argument("--output", "-o", default=None,
                        help="Arquivo NIfTI de saída")
    parser.add_argument("--threshold-min", type=int, default=200,
                        help="HU mínimo para osso (padrão: 200)")
    parser.add_argument("--threshold-max", type=int, default=3000,
                        help="HU máximo para osso (padrão: 3000)")
    parser.add_argument("--diff-threshold", type=float, default=None,
                        help="Limiar NCC de assimetria 0–1 (padrão: auto)")
    parser.add_argument("--n-sigma", type=float, default=3.0,
                        help="Para auto-threshold: média + N*sigma (padrão: 3.0)")
    parser.add_argument("--min-tamanho", type=int, default=500,
                        help="Tamanho mínimo de lesão em voxels (padrão: 500)")
    parser.add_argument("--espessura-casca", type=int, default=4,
                        help="Espessura da casca óssea em px (padrão: 4)")
    parser.add_argument("--tamanho-patch", type=int, default=16,
                        help="Tamanho do patch NCC em px (padrão: 16)")
    parser.add_argument("--sem-correcao-tilt", action="store_true",
                        help="Desativar correção de inclinação (PCA)")
    parser.add_argument("--salvar-mapa", action="store_true",
                        help="Salvar mapa bruto de assimetria NCC para inspeção")
    parser.add_argument("--dataset", action="store_true",
                        help="Processar todo o dataset")

    args = parser.parse_args()

    SCRIPT_DIR    = Path(__file__).parent.resolve()      # src/deteccao_lesao/
    DATASET_NIFTI = SCRIPT_DIR.parent.parent / "data" / "dataset_nifti"  # modelo_ia/data/

    kwargs = dict(
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        diff_threshold=args.diff_threshold,
        n_sigma=args.n_sigma,
        min_tamanho_lesao=args.min_tamanho,
        espessura_casca=args.espessura_casca,
        tamanho_patch=args.tamanho_patch,
        corrigir_tilt=not args.sem_correcao_tilt,
        salvar_mapa=args.salvar_mapa,
    )

    if args.dataset or args.input is None:
        processar_dataset(str(DATASET_NIFTI), **kwargs)
    else:
        detectar_lesao(args.input, output_path=args.output, **kwargs)
