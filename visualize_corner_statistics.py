#!/usr/bin/env python3
"""
Script pour visualiser les statistiques d'erreur par coin ChArUco
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
        return None, None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None, None
    
    return image, charuco_corners, charuco_ids


def project_points_with_cdist(obj_points, rvec, tvec, camera_matrix, dist_coeffs, distortion_center=None):
    """
    Projette les points 3D avec support de PP ≠ CDist
    """
    # S'assurer que rvec et tvec sont des vecteurs 1D
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


def visualize_corner_statistics(calibration_json, images_folder, square_size_cm,
                                 squares_x=11, squares_y=8, marker_ratio=0.7,
                                 output_file=None):
    """Visualise les statistiques d'erreur par coin ChArUco"""
    
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
    ], dtype=np.float64)
    
    # Extraire les coefficients de distorsion
    dist_coeffs_raw = calib_data['distortion_coefficients']
    dist_coeffs = np.array(dist_coeffs_raw, dtype=np.float64)
    if dist_coeffs.ndim > 1:
        dist_coeffs = dist_coeffs[0] if dist_coeffs.shape[0] == 1 else dist_coeffs.flatten()
    else:
        dist_coeffs = dist_coeffs.flatten()
    
    # Lire distortion_center si disponible
    distortion_center = calib_data.get('distortion_center', None)
    if distortion_center is None:
        distortion_center = calib_data['principal_point']
    
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
    
    # Obtenir les points 3D de la planche
    obj_points = board.getChessboardCorners()
    
    # Trouver les images
    images_folder = Path(images_folder)
    image_files = list(images_folder.glob("*.jpg")) + \
                  list(images_folder.glob("*.JPG")) + \
                  list(images_folder.glob("*.png")) + \
                  list(images_folder.glob("*.PNG")) + \
                  list(images_folder.glob("*.tiff")) + \
                  list(images_folder.glob("*.TIFF")) + \
                  list(images_folder.glob("*.tif")) + \
                  list(images_folder.glob("*.TIF"))
    
    if not image_files:
        raise ValueError(f"Aucune image trouvée dans {images_folder}")
    
    print(f"Traitement de {len(image_files)} images...")
    
    # Collecter les erreurs par coin
    errors_by_corner_id = defaultdict(list)
    valid_images = []
    
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
    
    # Traiter chaque image
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] {image_path.name}...")
        
        # Détecter les coins
        image, corners, ids = detect_charuco_corners(image_path, board)
        if corners is None or ids is None:
            print(f"  ⚠ Pas de coins détectés")
            continue
        
        corners_flat = corners.reshape(-1, 2).astype(np.float32)
        ids_flat = ids.flatten()
        
        # Obtenir les points 3D correspondants
        obj_pts = []
        for corner_id in ids_flat:
            if 0 <= corner_id < len(obj_points):
                obj_pts.append(obj_points[corner_id])
        
        if len(obj_pts) < 4:
            print(f"  ⚠ Pas assez de points ({len(obj_pts)} < 4)")
            continue
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        
        # Obtenir la pose (sauvegardée ou calculée)
        rvec = None
        tvec = None
        
        if has_saved_poses:
            image_path_str = str(image_path)
            try:
                saved_idx = saved_image_paths.index(image_path_str)
                rvec = saved_rvecs[saved_idx]
                tvec = saved_tvecs[saved_idx]
            except (ValueError, IndexError):
                # Calculer la pose avec solvePnP
                img_pts = corners_flat.reshape(-1, 1, 2)
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    camera_matrix,
                    dist_coeffs
                )
                if not success:
                    print(f"  ⚠ Échec de l'estimation de pose")
                    continue
        else:
            # Calculer la pose avec solvePnP
            img_pts = corners_flat.reshape(-1, 1, 2)
            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                camera_matrix,
                dist_coeffs
            )
            if not success:
                print(f"  ⚠ Échec de l'estimation de pose")
                continue
        
        # Projeter les points 3D sur l'image
        projected_points = project_points_with_cdist(
            obj_pts,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            distortion_center
        )
        
        # Calculer les erreurs (comme dans l'ancien script)
        # Note: corners_flat contient tous les coins, mais projected_points seulement ceux valides
        # On doit filtrer corners_flat pour ne garder que les coins valides
        valid_corners = []
        valid_ids = []
        for i, corner_id in enumerate(ids_flat):
            if 0 <= corner_id < len(obj_points):
                valid_corners.append(corners_flat[i])
                valid_ids.append(corner_id)
        
        if len(valid_corners) != len(projected_points):
            print(f"  ⚠ Incohérence: {len(valid_corners)} coins valides vs {len(projected_points)} projections")
            continue
        
        valid_corners = np.array(valid_corners)
        errors = np.linalg.norm(valid_corners - projected_points, axis=1)
        
        # Grouper par ID de coin
        for corner_id, err in zip(valid_ids, errors):
            errors_by_corner_id[corner_id].append(err)
        
        valid_images.append(image_path.name)
        print(f"  ✓ {len(corners_flat)} coins, erreur moyenne: {np.mean(errors):.3f} px")
    
    if not errors_by_corner_id:
        raise ValueError("Aucune erreur collectée")
    
    print(f"\n✓ {len(valid_images)} images valides, {len(errors_by_corner_id)} coins uniques")
    
    # Calculer les statistiques par coin
    corner_stats = {}
    all_errors_flat = []
    
    for corner_id, error_list in errors_by_corner_id.items():
        error_array = np.array(error_list)
        all_errors_flat.extend(error_list)
        
        # Position dans le damier (convertir de mètres en cm)
        if 0 <= corner_id < len(obj_points):
            board_position = obj_points[corner_id]
            board_x = float(board_position[0] * 100.0)  # mètres → cm
            board_y = float(board_position[1] * 100.0)  # mètres → cm
        else:
            board_x = 0.0
            board_y = 0.0
        
        corner_stats[corner_id] = {
            'mean_error': float(np.mean(error_array)),
            'median_error': float(np.median(error_array)),
            'max_error': float(np.max(error_array)),
            'min_error': float(np.min(error_array)),
            'std_error': float(np.std(error_array)),
            'count': len(error_list),
            'board_position': [board_x, board_y]
        }
    
    all_errors_flat = np.array(all_errors_flat)
    
    # Créer la visualisation
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    # Calculer les limites
    board_x_coords = [stats['board_position'][0] for stats in corner_stats.values()]
    board_y_coords = [stats['board_position'][1] for stats in corner_stats.values()]
    board_x_min = min(board_x_coords) if board_x_coords else 0
    board_x_max = max(board_x_coords) if board_x_coords else squares_x * square_size_cm
    board_y_min = min(board_y_coords) if board_y_coords else 0
    board_y_max = max(board_y_coords) if board_y_coords else squares_y * square_size_cm
    
    # Grille de référence
    grid_step = square_size_cm
    for x in np.arange(board_x_min, board_x_max + grid_step, grid_step):
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
    for y in np.arange(board_y_min, board_y_max + grid_step, grid_step):
        ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3)
    
    # Échelle de couleur basée sur l'erreur médiane
    median_errors = [stats['median_error'] for stats in corner_stats.values()]
    min_error = np.min(median_errors) if median_errors else 0
    max_error = np.percentile(median_errors, 95) if median_errors else 1
    error_range = max_error - min_error if max_error > min_error else 1.0
    
    # Colormap
    try:
        cmap = plt.colormaps['RdYlGn_r']
    except AttributeError:
        cmap = plt.cm.get_cmap('RdYlGn_r')
    
    # Taille des marqueurs basée sur l'écart-type
    std_errors = [stats['std_error'] for stats in corner_stats.values()]
    min_std = np.min(std_errors) if std_errors else 0
    max_std = np.max(std_errors) if std_errors else 1
    std_range = max_std - min_std if max_std > min_std else 1.0
    
    # Dessiner chaque coin
    for corner_id, stats in corner_stats.items():
        pos = stats['board_position']
        median_err = stats['median_error']
        std_err = stats['std_error']
        max_err = stats['max_error']
        count = stats['count']
        
        # Couleur basée sur l'erreur médiane
        norm_err = (median_err - min_error) / error_range if error_range > 0 else 0
        norm_err = np.clip(norm_err, 0, 1)
        color = cmap(norm_err)
        
        # Taille basée sur l'écart-type
        norm_std = (std_err - min_std) / std_range if std_range > 0 else 0
        norm_std = np.clip(norm_std, 0, 1)
        marker_size = 100 + (norm_std * 400)
        
        ax.scatter(pos[0], pos[1], s=marker_size, c=[color], 
                  edgecolors='white', linewidths=1.5, alpha=0.8, zorder=5)
        
        # Texte avec statistiques
        ax.text(pos[0] + square_size_cm * 0.1, pos[1] - square_size_cm * 0.1,
               f'ID:{corner_id}\nn:{count}\nmed:{median_err:.2f}\nstd:{std_err:.2f}\nmax:{max_err:.2f}',
               color='white', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
               ha='left', va='top')
    
    # Configuration des axes
    margin = square_size_cm * 0.5
    ax.set_xlim(board_x_min - margin, board_x_max + margin)
    ax.set_ylim(board_y_max + margin, board_y_min - margin)  # Inverser Y
    ax.set_xlabel('Colonne (cm)', fontsize=12, color='white')
    ax.set_ylabel('Ligne (cm)', fontsize=12, color='white')
    ax.tick_params(colors='white')
    
    # Titre
    global_mean = np.mean(all_errors_flat)
    global_median = np.median(all_errors_flat)
    global_max = np.max(all_errors_flat)
    global_std = np.std(all_errors_flat)
    
    title = (f"Statistiques d'erreur par coin ChArUco\n"
             f"{len(valid_images)} images, {len(corner_stats)} coins | "
             f"Moyenne: {global_mean:.2f} px | Médiane: {global_median:.2f} px | "
             f"Max: {global_max:.2f} px | σ: {global_std:.2f} px")
    ax.set_title(title, fontsize=12, color='white', pad=15)
    
    # Colorbar avec valeurs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_error, vmax=max_error))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Erreur médiane (pixels)', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white', labelsize=9)
    
    # Forcer l'affichage des valeurs numériques
    num_ticks = 5
    tick_values = np.linspace(min_error, max_error, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val:.3f}' for val in tick_values])
    # Forcer la mise à jour
    cbar.update_ticks()
    
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if output_file:
        plt.savefig(output_file, dpi=150, facecolor='black')
        print(f"\n✓ Figure sauvegardée: {output_file}")
    
    plt.show()
    
    # Statistiques dans la console
    print(f"\n{'='*60}")
    print(f"STATISTIQUES GLOBALES")
    print(f"{'='*60}")
    print(f"Images valides: {len(valid_images)}")
    print(f"Coins uniques: {len(corner_stats)}")
    print(f"Erreur moyenne: {global_mean:.4f} px")
    print(f"Erreur médiane: {global_median:.4f} px")
    print(f"Erreur std: {global_std:.4f} px")
    print(f"Erreur min: {np.min(all_errors_flat):.4f} px")
    print(f"Erreur max: {global_max:.4f} px")
    
    # Top 5 coins avec les plus grandes erreurs
    sorted_corners = sorted(corner_stats.items(), key=lambda x: x[1]['median_error'], reverse=True)
    print(f"\nTop 5 coins avec les erreurs médianes les plus élevées:")
    for i, (corner_id, stats) in enumerate(sorted_corners[:5], 1):
        print(f"  {i}. Coin ID {corner_id}: médiane={stats['median_error']:.4f} px, "
              f"moyenne={stats['mean_error']:.4f} px, max={stats['max_error']:.4f} px, "
              f"détecté {stats['count']} fois")


def main():
    parser = argparse.ArgumentParser(description='Visualise les statistiques d\'erreur par coin ChArUco')
    parser.add_argument('calibration_json', help='Fichier JSON de calibration')
    parser.add_argument('images_folder', help='Dossier contenant les images')
    parser.add_argument('--square-size', type=float, required=True,
                       help='Taille des carrés en cm')
    parser.add_argument('--squares-x', type=int, default=11,
                       help='Nombre de carrés en X (défaut: 11)')
    parser.add_argument('--squares-y', type=int, default=8,
                       help='Nombre de carrés en Y (défaut: 8)')
    parser.add_argument('--marker-ratio', type=float, default=0.7,
                       help='Ratio taille marqueur/carré (défaut: 0.7)')
    parser.add_argument('--output', type=str,
                       help='Fichier de sortie pour sauvegarder la visualisation (optionnel)')
    
    args = parser.parse_args()
    
    visualize_corner_statistics(
        args.calibration_json,
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio,
        args.output
    )


if __name__ == "__main__":
    main()

