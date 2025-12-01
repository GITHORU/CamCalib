#!/usr/bin/env python3
"""
Visualise les erreurs de reprojection pour chaque image individuellement
Affiche chaque coin détecté avec son erreur de projection
"""

import cv2
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


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


def visualize_image_errors(calibration_json, images_folder, square_size_cm,
                           squares_x=11, squares_y=8, marker_ratio=0.7,
                           output_folder=None, max_images=None, image_list=None,
                           exaggeration_factor=10, show_text=True):
    """Visualise les erreurs de reprojection pour chaque image individuellement"""
    
    # Charger la calibration
    with open(calibration_json, 'r') as f:
        calib_data = json.load(f)
    
    # Reconstruire camera_matrix à partir de focal_length et principal_point
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
    dist_coeffs = np.array(calib_data['distortion_coefficients'], dtype=np.float64)
    
    # Obtenir la taille de l'image
    image_size = tuple(calib_data['image_size'])
    
    # Créer la planche ChArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size_cm,
        square_size_cm * marker_ratio,
        aruco_dict
    )
    
    # Obtenir les points 3D de la planche
    obj_points = board.getChessboardCorners()
    
    # Vérifier si les poses et coins sont sauvegardés
    has_saved_poses = 'rvecs' in calib_data and 'tvecs' in calib_data and 'valid_image_paths' in calib_data
    has_saved_corners = 'detected_corners' in calib_data and 'detected_ids' in calib_data
    
    saved_rvecs = None
    saved_tvecs = None
    saved_image_paths = None
    saved_corners = None
    saved_ids = None
    
    if has_saved_poses:
        saved_rvecs = [np.array(rvec) for rvec in calib_data['rvecs']]
        saved_tvecs = [np.array(tvec) for tvec in calib_data['tvecs']]
        saved_image_paths = calib_data['valid_image_paths']
        print("✓ Utilisation des poses sauvegardées")
    
    if has_saved_corners:
        saved_corners = calib_data['detected_corners']
        saved_ids = calib_data['detected_ids']
        print("✓ Utilisation des coins sauvegardés")
    
    # Trouver les images à traiter
    images_folder = Path(images_folder)
    
    if image_list:
        # Utiliser la liste fournie
        image_files = []
        seen_paths = set()
        for img_name in image_list:
            img_path = Path(img_name)
            if not img_path.is_absolute():
                img_path = images_folder / img_name
            if img_path.exists():
                abs_path = img_path.resolve()
                if abs_path not in seen_paths:
                    seen_paths.add(abs_path)
                    image_files.append(img_path)
        
        if not image_files:
            raise ValueError("Aucune image valide trouvée dans la liste fournie")
    else:
        # Trouver toutes les images dans le dossier
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
        
        # Dédupliquer les images (par chemin absolu)
        seen_paths = set()
        unique_image_files = []
        for img_path in image_files:
            abs_path = img_path.resolve()
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                unique_image_files.append(img_path)
        
        image_files = unique_image_files
        
        # Limiter le nombre d'images si spécifié
        if max_images:
            image_files = image_files[:max_images]
    
    print(f"Traitement de {len(image_files)} images...")
    
    # Créer le dossier de sortie
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path("per_image_errors")
        output_path.mkdir(exist_ok=True)
    
    # Traiter chaque image et collecter les erreurs par coin
    from collections import defaultdict
    errors_by_corner_id = defaultdict(list)  # {corner_id: [liste des erreurs]}
    corner_positions = defaultdict(list)  # {corner_id: [liste des positions sur le capteur]}
    valid_images = []
    corners_to_save = []  # Pour sauvegarder les coins détectés si nécessaire
    ids_to_save = []
    
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Traitement de {image_path.name}...")
        
        # Charger l'image pour l'affichage
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ⚠ Impossible de charger l'image, ignoré")
            continue
        
        # Essayer d'utiliser les coins sauvegardés
        corners = None
        ids = None
        rvec = None
        tvec = None
        
        if has_saved_corners and has_saved_poses:
            # Chercher l'index de cette image dans la liste sauvegardée
            image_path_str = str(image_path)
            try:
                saved_idx = saved_image_paths.index(image_path_str)
                if saved_corners[saved_idx] is not None and saved_ids[saved_idx] is not None:
                    corners = np.array(saved_corners[saved_idx], dtype=np.float32)
                    ids = np.array(saved_ids[saved_idx], dtype=np.int32)
                    rvec = saved_rvecs[saved_idx]
                    tvec = saved_tvecs[saved_idx]
                    print(f"  ✓ Utilisation des coins et pose sauvegardés")
            except (ValueError, IndexError):
                # Image non trouvée dans la liste sauvegardée, détecter
                pass
        
        # Si pas de coins sauvegardés, détecter
        if corners is None or ids is None:
            _, corners, ids = detect_charuco_corners(image_path, board)
            if corners is None or ids is None:
                print(f"  ⚠ Pas de coins détectés, ignoré")
                corners_to_save.append(None)
                ids_to_save.append(None)
                continue
            
            # Sauvegarder pour la prochaine fois
            corners_to_save.append(corners.tolist())
            ids_to_save.append(ids.tolist())
            
            # Si pas de pose sauvegardée, calculer avec solvePnP
            if rvec is None or tvec is None:
                corners_flat = corners.reshape(-1, 2).astype(np.float32)
                ids_flat = ids.flatten()
                
                # Obtenir les points 3D correspondants
                obj_pts = []
                img_pts = []
                for i, corner_id in enumerate(ids_flat):
                    if 0 <= corner_id < len(obj_points):
                        obj_pts.append(obj_points[corner_id])
                        img_pts.append(corners_flat[i])
                
                if len(obj_pts) < 4:
                    print(f"  ⚠ Pas assez de points correspondants ({len(obj_pts)} < 4), ignoré")
                    continue
                
                obj_pts = np.array(obj_pts, dtype=np.float32)
                img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
                
                # Estimer la pose de la caméra
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    camera_matrix,
                    dist_coeffs
                )
                
                if not success:
                    print(f"  ⚠ Échec de l'estimation de pose, ignoré")
                    continue
        else:
            # Coins sauvegardés utilisés, pas besoin de les sauvegarder à nouveau
            corners_to_save.append(None)
            ids_to_save.append(None)
        
        # Convertir les coins et IDs au bon format
        corners_flat = corners.reshape(-1, 2).astype(np.float32)
        ids_flat = ids.flatten()
        
        # Obtenir les points 3D correspondants
        obj_pts = []
        for corner_id in ids_flat:
            if 0 <= corner_id < len(obj_points):
                obj_pts.append(obj_points[corner_id])
        
        if len(obj_pts) < 4:
            print(f"  ⚠ Pas assez de points correspondants ({len(obj_pts)} < 4), ignoré")
            continue
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        
        # Projeter les points 3D sur l'image
        projected_points, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculer les erreurs de reprojection
        errors = np.linalg.norm(corners_flat - projected_points, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        valid_images.append(image_path.name)
        
        print(f"  ✓ {len(corners_flat)} coins détectés")
        print(f"  ✓ Erreur moyenne: {mean_error:.4f} px")
        print(f"  ✓ Erreur max: {max_error:.4f} px")
        
        # Grouper les erreurs par ID de coin
        for corner_id, err, det_pt in zip(ids_flat, errors, corners_flat):
            errors_by_corner_id[corner_id].append(err)
            corner_positions[corner_id].append(det_pt)
    
    # Calculer les statistiques par coin
    if not errors_by_corner_id:
        print("\n⚠ Aucune erreur collectée")
        return output_path
    
    # Calculer les statistiques pour chaque coin
    corner_stats = {}
    all_errors_flat = []
    for corner_id, error_list in errors_by_corner_id.items():
        error_array = np.array(error_list)
        
        # Utiliser la position dans le damier (coordonnées 3D) au lieu de la position sur le capteur
        if 0 <= corner_id < len(obj_points):
            board_position = obj_points[corner_id]  # Position dans le damier (X, Y, Z) en cm
            board_x = float(board_position[0])
            board_y = float(board_position[1])
        else:
            # Si l'ID n'est pas valide, utiliser une position par défaut
            board_x = 0.0
            board_y = 0.0
        
        corner_stats[corner_id] = {
            'mean_error': float(np.mean(error_array)),
            'max_error': float(np.max(error_array)),
            'min_error': float(np.min(error_array)),
            'std_error': float(np.std(error_array)),
            'median_error': float(np.median(error_array)),
            'count': len(error_list),
            'board_position': [board_x, board_y]  # Position dans le damier en cm
        }
        all_errors_flat.extend(error_list)
    
    all_errors_flat = np.array(all_errors_flat)
    
    # Créer la visualisation globale par coin
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    # Calculer les dimensions du damier pour les limites des axes
    board_x_coords = [stats['board_position'][0] for stats in corner_stats.values()]
    board_y_coords = [stats['board_position'][1] for stats in corner_stats.values()]
    board_x_min = min(board_x_coords) if board_x_coords else 0
    board_x_max = max(board_x_coords) if board_x_coords else squares_x * square_size_cm
    board_y_min = min(board_y_coords) if board_y_coords else 0
    board_y_max = max(board_y_coords) if board_y_coords else squares_y * square_size_cm
    
    # Dessiner une grille pour référence (grille du damier)
    grid_step = square_size_cm
    for x in np.arange(board_x_min, board_x_max + grid_step, grid_step):
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
    for y in np.arange(board_y_min, board_y_max + grid_step, grid_step):
        ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3)
    
    # Définir les limites de l'échelle de couleur basée sur la médiane
    median_errors = [stats['median_error'] for stats in corner_stats.values()]
    min_error = np.min(median_errors) if median_errors else 0
    max_error = np.percentile(median_errors, 95) if median_errors else 1
    error_range = max_error - min_error if max_error > min_error else 1.0
    
    # Colormap pour les erreurs
    try:
        cmap = plt.colormaps['RdYlGn_r']
    except AttributeError:
        cmap = plt.cm.get_cmap('RdYlGn_r')
    
    # Calculer les limites pour la taille (basée sur std)
    std_errors = [stats['std_error'] for stats in corner_stats.values()]
    min_std = np.min(std_errors) if std_errors else 0
    max_std = np.max(std_errors) if std_errors else 1
    std_range = max_std - min_std if max_std > min_std else 1.0
    
    # Dessiner chaque coin avec son erreur médiane
    for corner_id, stats in corner_stats.items():
        pos = stats['board_position']  # Position dans le damier (cm)
        median_err = stats['median_error']
        std_err = stats['std_error']
        max_err = stats['max_error']
        count = stats['count']
        
        # Normaliser l'erreur médiane pour la couleur
        norm_err = (median_err - min_error) / error_range if error_range > 0 else 0
        norm_err = np.clip(norm_err, 0, 1)
        color = cmap(norm_err)
        
        # Dessiner le coin avec une taille proportionnelle à l'écart-type
        # Normaliser std pour la taille (entre 100 et 500)
        norm_std = (std_err - min_std) / std_range if std_range > 0 else 0
        norm_std = np.clip(norm_std, 0, 1)
        marker_size = 100 + (norm_std * 400)  # Taille variable selon l'écart-type (entre 100 et 500)
        ax.scatter(pos[0], pos[1], s=marker_size, c=[color], 
                  edgecolors='white', linewidths=1.5, alpha=0.8, zorder=5)
        
        # Afficher les statistiques en texte
        if show_text:
            std_err = stats['std_error']
            ax.text(pos[0] + square_size_cm * 0.1, pos[1] - square_size_cm * 0.1, 
                   f'ID:{corner_id}\nn:{count}\nmed:{median_err:.2f}\nstd:{std_err:.2f}\nmax:{max_err:.2f}',
                   color='white', fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                   ha='left', va='top')
    
    # Configuration de l'axe avec les dimensions du damier
    margin = square_size_cm * 0.5
    ax.set_xlim(board_x_min - margin, board_x_max + margin)
    ax.set_ylim(board_y_max + margin, board_y_min - margin)  # Inverser l'axe Y pour avoir l'origine en haut à gauche
    ax.set_xlabel('Colonne (cm)', fontsize=12, color='white')
    ax.set_ylabel('Ligne (cm)', fontsize=12, color='white')
    ax.tick_params(colors='white')
    
    # Titre avec statistiques
    global_mean = np.mean(all_errors_flat)
    global_median = np.median(all_errors_flat)
    global_max = np.max(all_errors_flat)
    global_std = np.std(all_errors_flat)
    
    title = (f"Résumé des erreurs par coin (exagération visuelle)\n"
             f"{len(valid_images)} images, {len(corner_stats)} coins uniques | "
             f"Erreur moyenne globale: {global_mean:.2f} px | "
             f"Médiane: {global_median:.2f} px | "
             f"Max: {global_max:.2f} px | "
             f"σ: {global_std:.2f} px")
    ax.set_title(title, fontsize=12, color='white', pad=15)
    
    # Ajouter la colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_error, vmax=max_error))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Erreur médiane (pixels)', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')
    
    # Afficher la figure de manière interactive
    plt.tight_layout()
    print(f"\n✓ Visualisation prête - fenêtre interactive ouverte")
    plt.show()
    
    # Statistiques globales
    print(f"\n{'='*60}")
    print(f"STATISTIQUES GLOBALES")
    print(f"{'='*60}")
    print(f"Images valides: {len(valid_images)}")
    print(f"Coins uniques: {len(corner_stats)}")
    print(f"Coins totaux: {len(all_errors_flat)}")
    print(f"Erreur moyenne globale: {global_mean:.4f} px")
    print(f"Erreur médiane: {global_median:.4f} px")
    print(f"Erreur std: {global_std:.4f} px")
    print(f"Erreur min: {np.min(all_errors_flat):.4f} px")
    print(f"Erreur max: {global_max:.4f} px")
    
    # Top 5 coins avec les plus grandes erreurs (basé sur la médiane)
    sorted_corners = sorted(corner_stats.items(), key=lambda x: x[1]['median_error'], reverse=True)
    print(f"\nTop 5 coins avec les erreurs médianes les plus élevées:")
    for i, (corner_id, stats) in enumerate(sorted_corners[:5], 1):
        print(f"  {i}. Coin ID {corner_id}: médiane={stats['median_error']:.4f} px, "
              f"moyenne={stats['mean_error']:.4f} px, max={stats['max_error']:.4f} px, détecté {stats['count']} fois")
    
    print(f"\nVisualisations sauvegardées dans: {output_path}")
    
    # Sauvegarder les coins détectés dans le fichier de calibration si nécessaire
    if corners_to_save and any(c is not None for c in corners_to_save):
        # Charger le fichier de calibration
        with open(calibration_json, 'r') as f:
            calib_data_to_update = json.load(f)
        
        # Mettre à jour les coins détectés
        # Si des coins existent déjà, les fusionner intelligemment
        if 'detected_corners' not in calib_data_to_update:
            calib_data_to_update['detected_corners'] = []
            calib_data_to_update['detected_ids'] = []
        
        # Créer un mapping des chemins d'images pour mettre à jour les bons indices
        if 'valid_image_paths' in calib_data_to_update:
            for idx, img_path in enumerate(image_files):
                img_path_str = str(img_path)
                try:
                    saved_idx = calib_data_to_update['valid_image_paths'].index(img_path_str)
                    if corners_to_save[idx] is not None:
                        # Mettre à jour les coins pour cette image
                        if saved_idx < len(calib_data_to_update['detected_corners']):
                            calib_data_to_update['detected_corners'][saved_idx] = corners_to_save[idx]
                            calib_data_to_update['detected_ids'][saved_idx] = ids_to_save[idx]
                        else:
                            # Ajouter à la fin si l'index n'existe pas encore
                            while len(calib_data_to_update['detected_corners']) < saved_idx:
                                calib_data_to_update['detected_corners'].append(None)
                                calib_data_to_update['detected_ids'].append(None)
                            calib_data_to_update['detected_corners'].append(corners_to_save[idx])
                            calib_data_to_update['detected_ids'].append(ids_to_save[idx])
                except ValueError:
                    # Image non dans la liste originale, l'ajouter
                    calib_data_to_update['valid_image_paths'].append(img_path_str)
                    calib_data_to_update['detected_corners'].append(corners_to_save[idx])
                    calib_data_to_update['detected_ids'].append(ids_to_save[idx])
        
        # Sauvegarder le fichier mis à jour
        with open(calibration_json, 'w') as f:
            json.dump(calib_data_to_update, f, indent=2)
        
        print(f"\n✓ Coins détectés sauvegardés dans {calibration_json}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualise les erreurs de reprojection pour chaque image individuellement'
    )
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
                       help='Dossier de sortie pour les visualisations (defaut: per_image_errors)')
    parser.add_argument('--max-images', type=int,
                       help='Nombre maximum d images a traiter')
    parser.add_argument('--image-list', type=str, nargs='+',
                       help='Liste specifique d images a analyser')
    parser.add_argument('--exaggeration', type=float, default=10,
                       help='Facteur d exageration des vecteurs d erreur (defaut: 10)')
    parser.add_argument('--no-text', action='store_true',
                       help='Ne pas afficher les valeurs d erreur en texte sur les images')
    
    args = parser.parse_args()
    
    visualize_image_errors(
        args.calibration_json,
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio,
        args.output,
        args.max_images,
        args.image_list,
        args.exaggeration,
        not args.no_text
    )

