#!/usr/bin/env python3
"""
Analyse détaillée des erreurs de reprojection d'une calibration
Génère des statistiques et des graphiques d'analyse
"""

import cv2
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def detect_charuco_corners(image_path, board):
    """Détecte les coins ChArUco dans une image"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None
    
    return charuco_corners, charuco_ids


def project_points_with_cdist(obj_points, rvec, tvec, camera_matrix, dist_coeffs, distortion_center=None):
    """
    Projette les points 3D avec support de PP ≠ CDist
    
    Args:
        obj_points: Points 3D (N, 3)
        rvec: Vecteur de rotation (3,)
        tvec: Vecteur de translation (3,)
        camera_matrix: Matrice caméra [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        dist_coeffs: Coefficients de distorsion [k1, k2, p1, p2, k3]
        distortion_center: [cx_dist, cy_dist] ou None (utilise PP si None)
    
    Returns:
        Points projetés (N, 2)
    """
    # S'assurer que rvec et tvec sont des vecteurs 1D (peuvent être (3,1) depuis JSON)
    rvec = np.array(rvec).flatten()
    tvec = np.array(tvec).flatten()
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Si pas de CDist spécifié, utiliser PP (comportement standard OpenCV)
    if distortion_center is None:
        return cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)[0].reshape(-1, 2)
    
    cx_dist, cy_dist = distortion_center
    
    # Extraire les coefficients de distorsion
    # dist_coeffs peut être (1, 5) ou (5,), s'assurer d'avoir un tableau 1D
    if isinstance(dist_coeffs, np.ndarray):
        if dist_coeffs.ndim > 1:
            dist_coeffs_flat = dist_coeffs[0] if dist_coeffs.shape[0] == 1 else dist_coeffs.flatten()
        else:
            dist_coeffs_flat = dist_coeffs
    else:
        dist_coeffs_flat = np.array(dist_coeffs).flatten()
    
    k1 = float(dist_coeffs_flat[0]) if len(dist_coeffs_flat) > 0 else 0.0
    k2 = float(dist_coeffs_flat[1]) if len(dist_coeffs_flat) > 1 else 0.0
    k3 = float(dist_coeffs_flat[4]) if len(dist_coeffs_flat) > 4 else 0.0
    p1 = float(dist_coeffs_flat[2]) if len(dist_coeffs_flat) > 2 else 0.0
    p2 = float(dist_coeffs_flat[3]) if len(dist_coeffs_flat) > 3 else 0.0
    
    # Rotation et translation
    R, _ = cv2.Rodrigues(rvec)
    obj_points_rot = (R @ obj_points.T).T + tvec
    
    # Projection perspective avec PP
    x_proj = fx * obj_points_rot[:, 0] / obj_points_rot[:, 2] + cx
    y_proj = fy * obj_points_rot[:, 1] / obj_points_rot[:, 2] + cy
    
    # Normaliser par rapport à CDist (pas PP!)
    x_norm = (x_proj - cx_dist) / fx
    y_norm = (y_proj - cy_dist) / fy
    
    # Calculer r²
    r_squared = x_norm**2 + y_norm**2
    
    # Distorsion radiale
    radial = 1 + k1 * r_squared + k2 * r_squared**2 + k3 * r_squared**3
    
    # Distorsion tangentielle
    x_dist = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r_squared + 2 * x_norm**2)
    y_dist = y_norm * radial + 2 * p2 * x_norm * y_norm + p1 * (r_squared + 2 * y_norm**2)
    
    # Remettre dans le système image avec CDist
    x_final = x_dist * fx + cx_dist
    y_final = y_dist * fy + cy_dist
    
    return np.column_stack([x_final, y_final])


def analyze_calibration_errors(calibration_json, images_folder, square_size_cm,
                               squares_x=11, squares_y=8, marker_ratio=0.7,
                               output_folder=None, max_images=None, image_list=None,
                               error_min=None, error_max=None):
    """Analyse détaillée des erreurs de reprojection"""
    
    # Charger la calibration
    with open(calibration_json, 'r') as f:
        calib_data = json.load(f)
    
    # Reconstruire camera_matrix
    fx = calib_data['focal_length'][0]
    fy = calib_data['focal_length'][1]
    cx = calib_data['principal_point'][0]
    cy = calib_data['principal_point'][1]
    
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    
    # Extraire les coefficients de distorsion
    dist_coeffs_raw = calib_data['distortion_coefficients']
    dist_coeffs = np.array(dist_coeffs_raw, dtype=np.float64)
    # Gérer le format imbriqué (peut être [[k1, k2, ...]] ou [k1, k2, ...])
    if dist_coeffs.ndim > 1:
        dist_coeffs = dist_coeffs[0] if dist_coeffs.shape[0] == 1 else dist_coeffs.flatten()
    else:
        dist_coeffs = dist_coeffs.flatten()
    
    image_size = tuple(calib_data['image_size'])
    
    # Lire distortion_center si disponible (PP ≠ CDist), sinon utiliser PP
    distortion_center = calib_data.get('distortion_center', None)
    if distortion_center is None:
        distortion_center = calib_data['principal_point']  # Fallback sur PP
    
    print(f"=== ANALYSE DES ERREURS DE CALIBRATION ===")
    print(f"Calibration: {calibration_json}")
    print(f"Dossier images: {images_folder}")
    print(f"Erreur moyenne: {calib_data['reprojection_error']:.3f} pixels")
    print("=" * 50)
    
    # Créer la planche ChArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Convertir cm en mètres (OpenCV attend les dimensions en mètres)
    square_size_m = square_size_cm / 100.0
    marker_size_m = square_size_m * marker_ratio
    
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size_m,      # En mètres
        marker_size_m,      # En mètres
        aruco_dict
    )
    
    obj_points_all = board.getChessboardCorners()
    
    # Vérifier si les poses sont sauvegardées
    has_saved_poses = 'rvecs' in calib_data and 'tvecs' in calib_data and 'valid_image_paths' in calib_data
    saved_rvecs = None
    saved_tvecs = None
    saved_image_paths = None
    
    if has_saved_poses:
        saved_rvecs = [np.array(rvec).flatten() for rvec in calib_data['rvecs']]
        saved_tvecs = [np.array(tvec).flatten() for tvec in calib_data['tvecs']]
        saved_image_paths = calib_data['valid_image_paths']
        print("✓ Utilisation des poses sauvegardées")
    
    # Trouver les images
    if image_list:
        images_folder_path = Path(images_folder)
        image_files = []
        for img_path in image_list:
            img = Path(img_path)
            if not img.is_absolute():
                full_path = images_folder_path / img
                if full_path.exists():
                    image_files.append(full_path)
                elif img.exists():
                    image_files.append(img)
            else:
                if img.exists():
                    image_files.append(img)
    else:
        image_files = list(Path(images_folder).glob("*.jpg")) + \
                      list(Path(images_folder).glob("*.JPG")) + \
                      list(Path(images_folder).glob("*.png")) + \
                      list(Path(images_folder).glob("*.PNG")) + \
                      list(Path(images_folder).glob("*.tiff")) + \
                      list(Path(images_folder).glob("*.TIFF")) + \
                      list(Path(images_folder).glob("*.tif")) + \
                      list(Path(images_folder).glob("*.TIF"))
        
        if max_images:
            image_files = image_files[:max_images]
    
    # Dédupliquer les images (par chemin absolu pour éviter les doublons)
    seen_paths = set()
    unique_image_files = []
    for img_path in image_files:
        abs_path = img_path.resolve()  # Chemin absolu normalisé
        if abs_path not in seen_paths:
            seen_paths.add(abs_path)
            unique_image_files.append(img_path)
    
    image_files = unique_image_files
    print(f"Traitement de {len(image_files)} images (après déduplication)...")
    
    # Collecter les données d'erreur
    all_errors = []
    errors_by_image_dict = {}  # Dictionnaire pour grouper par nom d'image
    errors_by_position = []  # (x, y, error)
    
    processed = 0
    
    for idx, image_path in enumerate(image_files):
        corners, ids = detect_charuco_corners(image_path, board)
        
        if corners is None or ids is None or len(ids) < 4:
            continue
        
        ids_flat = ids.flatten() if isinstance(ids, np.ndarray) else ids
        if len(corners.shape) > 2:
            corners_flat = corners.reshape(-1, 2)
        else:
            corners_flat = corners
        
        obj_pts = []
        img_pts = []
        for i, corner_id in enumerate(ids_flat):
            try:
                corner_id_int = int(corner_id)
                if 0 <= corner_id_int < len(obj_points_all):
                    obj_pts.append(obj_points_all[corner_id_int])
                    img_pts.append(corners_flat[i])
            except (ValueError, IndexError, TypeError):
                continue
        
        if len(obj_pts) < 4:
            continue
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        
        # Utiliser la pose sauvegardée si disponible
        if has_saved_poses:
            image_path_str = str(image_path)
            try:
                saved_idx = saved_image_paths.index(image_path_str)
                rvec = saved_rvecs[saved_idx]
                tvec = saved_tvecs[saved_idx]
            except (ValueError, IndexError):
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
                if not success:
                    continue
        else:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
            if not success:
                continue
        
        # Projeter et calculer les erreurs
        # Utilise distortion_center si disponible (PP ≠ CDist)
        projected_pts = project_points_with_cdist(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs, distortion_center
        )
        detected_pts = img_pts.reshape(-1, 2)
        
        errors = np.linalg.norm(detected_pts - projected_pts, axis=1)
        
        # Collecter les données
        all_errors.extend(errors.tolist())
        
        # Grouper par nom d'image (au cas où il y aurait des doublons)
        img_name = image_path.name
        if img_name not in errors_by_image_dict:
            errors_by_image_dict[img_name] = {
                'name': img_name,
                'errors': [],
                'num_corners': 0
            }
        
        errors_by_image_dict[img_name]['errors'].extend(errors.tolist())
        errors_by_image_dict[img_name]['num_corners'] += len(errors)
        
        # Erreurs par position
        for det_pt, err in zip(detected_pts, errors):
            if 0 <= det_pt[0] < image_size[0] and 0 <= det_pt[1] < image_size[1]:
                errors_by_position.append((det_pt[0], det_pt[1], err))
        
        processed += 1
        
        if processed % 10 == 0:
            print(f"  Traité {processed}/{len(image_files)} images...")
    
    if not all_errors:
        raise ValueError("Aucune erreur calculée")
    
    # Calculer les statistiques par image (après regroupement)
    errors_by_image = []
    for img_name, img_data in errors_by_image_dict.items():
        img_errors = np.array(img_data['errors'])
        errors_by_image.append({
            'name': img_name,
            'mean_error': float(np.mean(img_errors)),
            'max_error': float(np.max(img_errors)),
            'min_error': float(np.min(img_errors)),
            'std_error': float(np.std(img_errors)),
            'num_corners': img_data['num_corners']
        })
    
    all_errors = np.array(all_errors)
    errors_by_position = np.array(errors_by_position)
    image_names = [img['name'] for img in errors_by_image]
    
    print(f"\n✓ {len(errors_by_image)} images uniques analysées, {len(all_errors)} coins")
    
    # Calculer les statistiques globales
    stats = {
        'mean': np.mean(all_errors),
        'median': np.median(all_errors),
        'std': np.std(all_errors),
        'min': np.min(all_errors),
        'max': np.max(all_errors),
        'percentile_25': np.percentile(all_errors, 25),
        'percentile_75': np.percentile(all_errors, 75),
        'percentile_95': np.percentile(all_errors, 95),
        'percentile_99': np.percentile(all_errors, 99)
    }
    
    # Définir les limites de l'échelle de couleur pour la heatmap
    if error_min is not None:
        heatmap_vmin = error_min
    else:
        heatmap_vmin = stats['min']
    
    if error_max is not None:
        heatmap_vmax = error_max
    else:
        heatmap_vmax = stats['percentile_95']  # Utiliser le 95e percentile pour éviter les outliers
    
    # Créer le dossier de sortie
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = Path("error_analysis")
        output_path.mkdir(exist_ok=True)
    
    # 1. Graphiques principaux
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Analyse des erreurs de reprojection', fontsize=16, fontweight='bold')
    
    # Histogramme
    ax1 = axes[0]
    ax1.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(stats['mean'], color='r', linestyle='--', label=f"Moyenne: {stats['mean']:.3f} px")
    ax1.axvline(stats['median'], color='g', linestyle='--', label=f"Médiane: {stats['median']:.3f} px")
    ax1.set_xlabel('Erreur (pixels)')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des erreurs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Erreurs par image
    ax2 = axes[1]
    image_means = [img['mean_error'] for img in errors_by_image]
    image_indices = range(len(image_means))
    ax2.plot(image_indices, image_means, 'o-', markersize=3, alpha=0.6)
    ax2.axhline(stats['mean'], color='r', linestyle='--', label=f"Moyenne globale")
    ax2.set_xlabel('Index de l\'image')
    ax2.set_ylabel('Erreur moyenne (pixels)')
    ax2.set_title('Erreur moyenne par image')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Carte de chaleur des erreurs par position
    ax3 = axes[2]
    if len(errors_by_position) > 0:
        # Créer une grille pour la carte de chaleur
        grid_size = 50
        x_bins = np.linspace(0, image_size[0], grid_size)
        y_bins = np.linspace(0, image_size[1], grid_size)
        
        # Calculer l'erreur moyenne dans chaque cellule
        heatmap = np.zeros((grid_size-1, grid_size-1))
        counts = np.zeros((grid_size-1, grid_size-1))
        
        for x, y, err in errors_by_position:
            i = np.digitize(x, x_bins) - 1
            j = np.digitize(y, y_bins) - 1
            if 0 <= i < grid_size-1 and 0 <= j < grid_size-1:
                heatmap[j, i] += err
                counts[j, i] += 1
        
        # Moyenne
        mask = counts > 0
        heatmap[mask] /= counts[mask]
        
        im = ax3.imshow(heatmap, origin='upper', extent=[0, image_size[0], image_size[1], 0],
                       cmap='hot', interpolation='bilinear', aspect='auto',
                       vmin=heatmap_vmin, vmax=heatmap_vmax)
        ax3.plot(cx, cy, 'x', color='cyan', markersize=15, markeredgewidth=2, label='PP')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        ax3.set_title('Carte de chaleur des erreurs')
        ax3.legend()
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Erreur moyenne (pixels)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Graphiques sauvegardés: {output_path / 'error_analysis.png'}")
    
    # 2. Sauvegarder les statistiques en JSON
    analysis_results = {
        'global_statistics': stats,
        'per_image_statistics': errors_by_image,
        'total_images': len(errors_by_image),
        'total_corners': len(all_errors)
    }
    
    with open(output_path / 'error_statistics.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"✓ Statistiques sauvegardées: {output_path / 'error_statistics.json'}")
    
    # 4. Afficher un résumé dans la console
    print("\n" + "=" * 50)
    print("RÉSUMÉ DES STATISTIQUES")
    print("=" * 50)
    print(f"Nombre d'images analysées: {processed}")
    print(f"Nombre total de coins: {len(all_errors)}")
    print(f"\nErreurs globales:")
    print(f"  Moyenne: {stats['mean']:.3f} pixels")
    print(f"  Médiane: {stats['median']:.3f} pixels")
    print(f"  Écart-type: {stats['std']:.3f} pixels")
    print(f"  Min: {stats['min']:.3f} pixels")
    print(f"  Max: {stats['max']:.3f} pixels")
    print(f"\nPercentiles:")
    print(f"  25e: {stats['percentile_25']:.3f} pixels")
    print(f"  75e: {stats['percentile_75']:.3f} pixels")
    print(f"  95e: {stats['percentile_95']:.3f} pixels")
    print(f"  99e: {stats['percentile_99']:.3f} pixels")
    
    # Images avec les erreurs les plus élevées
    sorted_images = sorted(errors_by_image, key=lambda x: x['mean_error'], reverse=True)
    print(f"\nTop 5 images avec les erreurs les plus élevées:")
    for i, img in enumerate(sorted_images[:5], 1):
        print(f"  {i}. {img['name']}: {img['mean_error']:.3f} px (max: {img['max_error']:.3f} px)")
    
    print(f"\n✓ Analyse complète sauvegardée dans: {output_path}")
    
    if not output_folder:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyse détaillée des erreurs de calibration')
    parser.add_argument('calibration_json', help='Fichier JSON de calibration')
    parser.add_argument('images_folder', help='Dossier contenant les images')
    parser.add_argument('--square-size', type=float, required=True,
                       help='Taille des carres en cm')
    parser.add_argument('--squares-x', type=int, default=11,
                       help='Nombre de carres en X (defaut: 11)')
    parser.add_argument('--squares-y', type=int, default=8,
                       help='Nombre de carres en Y (defaut: 8)')
    parser.add_argument('--marker-ratio', type=float, default=0.7,
                       help='Ratio taille marqueur/carre (defaut: 0.7)')
    parser.add_argument('--output', type=str,
                       help='Dossier de sortie pour les resultats (defaut: error_analysis)')
    parser.add_argument('--max-images', type=int,
                       help='Nombre maximum d images a traiter')
    parser.add_argument('--image-list', type=str, nargs='+',
                       help='Liste specifique d images a analyser')
    parser.add_argument('--error-min', type=float,
                       help='Valeur minimale pour l echelle de couleur de la heatmap (pixels)')
    parser.add_argument('--error-max', type=float,
                       help='Valeur maximale pour l echelle de couleur de la heatmap (pixels)')
    
    args = parser.parse_args()
    
    analyze_calibration_errors(
        args.calibration_json,
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio,
        args.output,
        args.max_images,
        args.image_list,
        args.error_min,
        args.error_max
    )


if __name__ == "__main__":
    main()

