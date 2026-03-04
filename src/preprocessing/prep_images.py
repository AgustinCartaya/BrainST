

import numpy as np



def update_affine(original_affine: np.ndarray, offset: tuple) -> np.ndarray:
    """
    Modifica la matriz affine de acuerdo al desplazamiento en voxels.
    
    :param original_affine: Matriz affine original (4x4)
    :param offset: Desplazamiento en voxels (dx, dy, dz)
    :return: Matriz affine ajustada
    """
    dx, dy, dz = offset
    translation = original_affine[:3, :3] @ np.array([dx, dy, dz])
    new_affine = original_affine.copy()
    new_affine[:3, 3] -= translation
    return new_affine


def resize_center_crop_pad(image: np.ndarray, new_shape: tuple, affine=None) -> np.ndarray:
    """
    Ajusta el tamaño de una imagen 3D (x, y, z) recortando o rellenando con ceros.
    
    :param image: np.ndarray de tamaño (x, y, z)
    :param new_shape: Tuple (nx, ny, nz) con las nuevas dimensiones
    :return: np.ndarray de tamaño (nx, ny, nz)
    """
    x, y, z = image.shape
    nx, ny, nz = new_shape
    
    # Inicializar imagen de salida con ceros
    new_image = np.zeros((nx, ny, nz), dtype=image.dtype)
    
    # Calcular los índices de recorte o padding para cada dimensión
    def get_slices(old, new):
        if old > new:
            start = (old - new) // 2
            return slice(start, start + new), slice(0, new)
        else:
            start = (new - old) // 2
            return slice(0, old), slice(start, start + old)
    
    x_slice_old, x_slice_new = get_slices(x, nx)
    y_slice_old, y_slice_new = get_slices(y, ny)
    z_slice_old, z_slice_new = get_slices(z, nz)
    
    # Copiar los datos ajustados
    new_image[x_slice_new, y_slice_new, z_slice_new] = image[x_slice_old, y_slice_old, z_slice_old]
    offset = (x_slice_new.start-x_slice_old.start, y_slice_new.start-y_slice_old.start, z_slice_new.start-z_slice_old.start)

    if affine is not None:
        if type(affine) is tuple:
            # Actualizar la matriz affine si se proporciona
            new_affine_matrix = update_affine(affine[0], offset)
            return new_image, offset, (new_affine_matrix, affine[1])
        else:
            new_affine_matrix = update_affine(affine, offset)
            return new_image, offset, new_affine_matrix
    else:
        return new_image, offset



def robust_normalize(
    img,
    percentile=(0, 100),
    mask=None,
    reference_tensor=None,
    strictly_positive=True,
    clip_values = True
):
    """
    Normaliza una imagen al rango [0, 1] usando percentiles robustos.
    Soporta máscaras, tensor de referencia y opción de valores estrictamente positivos.

    Parámetros:
        img (np.ndarray): imagen o tensor a normalizar.
        percentile (tuple): percentiles para la normalización (p_min, p_max). Por defecto (0.5, 99.5).
        mask (np.ndarray, opcional): máscara booleana o binaria para calcular percentiles solo en regiones válidas.
        reference_tensor (np.ndarray, opcional): tensor del cual se obtendrán los percentiles. Si es None, se usa `img`.
        strictly_positive (bool): si es True, se fuerza que el valor mínimo no sea menor a 0.

    Retorna:
        np.ndarray: imagen normalizada en el rango [0, 1].
    """

    # Determinar tensor de referencia
    ref = reference_tensor if reference_tensor is not None else img

    # Aplicar máscara si se proporciona
    if mask is not None:
        ref = ref[mask > 0]

    # Calcular percentiles
    p_min, p_max = np.percentile(ref, percentile)

    # Ajuste para valores estrictamente positivos
    if strictly_positive and p_min < 0:
        p_min = 0

    # Clip antes de normalizar
    if clip_values:
        img_clipped = np.clip(img, p_min, p_max)
    else:
        img_clipped = img

    # Normalización al rango [0, 1]
    if p_max > p_min:
        img_normalized = (img_clipped - p_min) / (p_max - p_min)
    else:
        img_normalized = np.zeros_like(img)

    return img_normalized




def preprocess_image(img, aff):
    img, offset, new_aff = resize_center_crop_pad(img, (192, 256, 192), aff)
    img = robust_normalize(img, percentile=(0,100), strictly_positive=True)
    return img, new_aff


def postprocess_image(img, org_shape):
    img, offset = resize_center_crop_pad(img, org_shape, None)
    return img