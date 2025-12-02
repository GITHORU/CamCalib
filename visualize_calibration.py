#!/usr/bin/env python3
"""
Visualise les erreurs de reprojection d'une calibration ChArUco
Affiche les vecteurs d'erreur entre les coins détectés et projetés
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
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None
    
    return charuco_corners, charuco_ids


def visualize_reprojection_errors(calibration_json, images_folder, square_size_cm, 
                                   squares_x=11, squares_y=8, marker_ratio=0.7,
                                   output_file=None, max_images=None, image_list=None,
                                   error_min=None, error_max=None, exaggeration_factor=10):
    """Visualise les erreurs de reprojection superposées sur une seule vue du capteur"""
    
    # Charger la calibration
    with open(calibration_json, 'r') as f:
        calib_data = json.load(f)
    
    # Reconstruire camera_matrix à partir de focal_length et principal_point
    # pour prendre en compte les modifications manuelles
    fx = calib_data['focal_length'][0]
    fy = calib_data['focal_length'][1]
    cx = calib_data['principal_point'][0]
    cy = calib_data['principal_point'][1]
    
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs = np.array(calib_data['distortion_coefficients'][0])
    image_size = tuple(calib_data['image_size'])
    
    print(f"=== VISUALISATION GLOBALE DES ERREURS ===")
    print(f"Calibration: {calibration_json}")
    print(f"Dossier images: {images_folder}")
    print(f"Erreur moyenne: {calib_data['reprojection_error']:.3f} pixels")
    print("=" * 50)
    
    # Créer la planche ChArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_size_cm,
        square_size_cm * marker_ratio,
        aruco_dict
    )
    
    # Obtenir les points 3D de la planche (tous les coins possibles)
    obj_points_all = board.getChessboardCorners()
    
    # Si une liste d'images est fournie, l'utiliser
    if image_list:
        # Résoudre les chemins (relatifs au dossier images_folder ou absolus)
        image_files = []
        images_folder_path = Path(images_folder)
        for img_path in image_list:
            img = Path(img_path)
            # Si le chemin est relatif, essayer depuis le dossier images_folder
            if not img.is_absolute():
                full_path = images_folder_path / img
                if full_path.exists():
                    image_files.append(full_path)
                elif img.exists():
                    image_files.append(img)
                else:
                    print(f"Attention: image non trouvée: {img_path}")
            else:
                if img.exists():
                    image_files.append(img)
                else:
                    print(f"Attention: image non trouvée: {img_path}")
        
        # Dédupliquer les images (par chemin absolu)
        seen_paths = set()
        unique_image_files = []
        for img_path in image_files:
            abs_path = img_path.resolve()
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                unique_image_files.append(img_path)
        
        image_files = unique_image_files
        print(f"Utilisation de la liste fournie: {len(image_files)} images valides (après déduplication)")
        if not image_files:
            raise ValueError("Aucune image valide trouvée dans la liste fournie")
    else:
        # Trouver toutes les images dans le dossier
        image_files = list(Path(images_folder).glob("*.jpg")) + \
                      list(Path(images_folder).glob("*.JPG")) + \
                      list(Path(images_folder).glob("*.png")) + \
                      list(Path(images_folder).glob("*.PNG")) + \
                      list(Path(images_folder).glob("*.tiff")) + \
                      list(Path(images_folder).glob("*.TIFF")) + \
                      list(Path(images_folder).glob("*.tif")) + \
                      list(Path(images_folder).glob("*.TIF"))
        
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
    
    print(f"Traitement de {len(image_files)} images (après déduplication)...")
    
    # Vérifier si les poses sont sauvegardées dans le fichier de calibration
    has_saved_poses = 'rvecs' in calib_data and 'tvecs' in calib_data and 'valid_image_paths' in calib_data
    saved_rvecs = None
    saved_tvecs = None
    saved_image_paths = None
    
    if has_saved_poses:
        saved_rvecs = [np.array(rvec) for rvec in calib_data['rvecs']]
        saved_tvecs = [np.array(tvec) for tvec in calib_data['tvecs']]
        saved_image_paths = calib_data['valid_image_paths']
        print("✓ Utilisation des poses sauvegardées (les changements de calibration seront visibles)")
    else:
        print("⚠ Poses non sauvegardées - recalcul avec solvePnP (les changements peuvent être masqués)")
    
    # Collecter tous les coins et leurs erreurs
    all_detected_pts = []
    all_projected_pts = []
    all_errors = []
    
    processed = 0
    
    for idx, image_path in enumerate(image_files):
        corners, ids = detect_charuco_corners(image_path, board)
        
        if corners is None or ids is None or len(ids) < 4:
            continue
        
        # Convertir les IDs et corners en format plat
        ids_flat = ids.flatten() if isinstance(ids, np.ndarray) else ids
        if len(corners.shape) > 2:
            corners_flat = corners.reshape(-1, 2)
        else:
            corners_flat = corners
        
        # Obtenir les points 3D correspondants aux IDs détectés
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
        
        # Utiliser la pose sauvegardée si disponible, sinon recalculer
        if has_saved_poses:
            # Trouver l'index de l'image dans la liste sauvegardée
            image_path_str = str(image_path)
            try:
                saved_idx = saved_image_paths.index(image_path_str)
                rvec = saved_rvecs[saved_idx]
                tvec = saved_tvecs[saved_idx]
            except (ValueError, IndexError):
                # Si l'image n'est pas dans la liste sauvegardée, recalculer
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs
                )
                if not success:
                    continue
        else:
            # Résoudre PnP pour obtenir la pose avec les nouveaux paramètres
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs
            )
            if not success:
                continue
        
        # Projeter les points 3D avec les paramètres de calibration (peuvent être différents)
        projected_pts, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_pts = projected_pts.reshape(-1, 2)
        
        # Calculer les erreurs
        detected_pts = img_pts.reshape(-1, 2)
        errors = np.linalg.norm(detected_pts - projected_pts, axis=1)
        
        # Ajouter à la collection globale
        all_detected_pts.extend(detected_pts.tolist())
        all_projected_pts.extend(projected_pts.tolist())
        all_errors.extend(errors.tolist())
        
        processed += 1
        if processed % 10 == 0:
            print(f"  Traité {processed}/{len(image_files)} images...")
    
    if not all_errors:
        raise ValueError("Aucun coin détecté dans les images")
    
    all_detected_pts = np.array(all_detected_pts)
    all_projected_pts = np.array(all_projected_pts)
    all_errors = np.array(all_errors)
    
    print(f"\nCoins totaux collectés: {len(all_errors)}")
    
    # Calculer les statistiques
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    max_error_actual = np.max(all_errors)
    min_error_actual = np.min(all_errors)
    std_error = np.std(all_errors)
    
    # Créer la figure matplotlib
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    
    # Dessiner une grille pour référence
    grid_step = 500
    for x in range(0, image_size[0] + grid_step, grid_step):
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
    for y in range(0, image_size[1] + grid_step, grid_step):
        ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3)
    
    # Dessiner le point principal
    cx, cy = calib_data['principal_point']
    ax.plot(cx, cy, 'x', color='white', markersize=15, markeredgewidth=2, label='Point principal')
    ax.plot([cx - 30, cx + 30], [cy, cy], 'w-', linewidth=1, alpha=0.5)
    ax.plot([cx, cx], [cy - 30, cy + 30], 'w-', linewidth=1, alpha=0.5)
    
    # Définir les limites de l'échelle de couleur
    if error_min is not None:
        min_error = error_min
    else:
        min_error = np.min(all_errors)
    
    if error_max is not None:
        max_error = error_max
    else:
        max_error = np.percentile(all_errors, 95)  # Utiliser le 95e percentile pour éviter les outliers
    
    error_range = max_error - min_error if max_error > min_error else 1.0
    
    # Filtrer les points dans les limites de l'image
    mask = (all_detected_pts[:, 0] >= 0) & (all_detected_pts[:, 0] < image_size[0]) & \
           (all_detected_pts[:, 1] >= 0) & (all_detected_pts[:, 1] < image_size[1])
    detected_filtered = all_detected_pts[mask]
    projected_filtered = all_projected_pts[mask]
    errors_filtered = all_errors[mask]
    
    # Créer une colormap pour les erreurs
    try:
        cmap = plt.colormaps['RdYlGn_r']
    except AttributeError:
        cmap = plt.cm.get_cmap('RdYlGn_r')
    
    # Normaliser les erreurs BRUTES pour le code couleur (sans exagération)
    normalized_errors = (errors_filtered - min_error) / error_range
    normalized_errors = np.clip(normalized_errors, 0, 1)
    
    # Exagération UNIQUEMENT de la longueur des flèches (facteur configurable)
    # Les valeurs d'erreur pour la couleur restent brutes
    error_vectors = projected_filtered - detected_filtered
    projected_exaggerated = detected_filtered + exaggeration_factor * error_vectors
    
    # Dessiner toutes les flèches avec code couleur selon l'erreur BRUTE
    for det_pt, proj_exag_pt, err, norm_err in zip(detected_filtered, 
                                                    projected_exaggerated,
                                                    errors_filtered,  # Erreur brute pour la couleur
                                                    normalized_errors):  # Erreur normalisée pour la couleur
        # Couleur selon l'erreur brute normalisée (sans exagération)
        color = cmap(norm_err)
        # Longueur de la flèche exagérée (x10)
        ax.annotate('', xy=proj_exag_pt, xytext=det_pt,
                   arrowprops=dict(arrowstyle='->', color=color, 
                                 alpha=0.7, lw=1.5))
    
    # Configuration de l'axe
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)  # Inverser l'axe Y pour correspondre aux coordonnées image
    ax.set_xlabel('X (pixels)', fontsize=12, color='white')
    ax.set_ylabel('Y (pixels)', fontsize=12, color='white')
    ax.tick_params(colors='white')
    
    # Titre avec statistiques
    title = (f"Vue globale des erreurs de reprojection (exagération x{exaggeration_factor})\n"
             f"{processed} images, {len(all_errors)} coins | "
             f"Erreur moyenne: {mean_error:.2f} px | "
             f"Médiane: {median_error:.2f} px | "
             f"Max: {max_error_actual:.2f} px | "
             f"σ: {std_error:.2f} px")
    ax.set_title(title, fontsize=11, color='white', pad=20)
    
    # Créer une colorbar avec un scatter invisible pour la légende
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_error, vmax=max_error))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Erreur (pixels)', color='white', fontsize=11)
    
    # Légende
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', 
             labelcolor='white', fontsize=10)
    
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if output_file:
        plt.savefig(output_file, dpi=150, facecolor='black', bbox_inches='tight')
        print(f"\n✓ Visualisation sauvegardée: {output_file}")
    else:
        plt.show()
    
    print(f"\n=== STATISTIQUES GLOBALES ===")
    print(f"Images traitées: {processed}")
    print(f"Coins totaux: {len(all_errors)}")
    print(f"Erreur moyenne: {mean_error:.3f} pixels")
    print(f"Erreur médiane: {median_error:.3f} pixels")
    print(f"Erreur max: {max_error_actual:.3f} pixels")
    print(f"Erreur min: {min_error_actual:.3f} pixels")
    print(f"Écart-type: {std_error:.3f} pixels")


def main():
    parser = argparse.ArgumentParser(description='Visualise les erreurs de reprojection')
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
                       help='Fichier de sortie pour sauvegarder la visualisation (optionnel)')
    parser.add_argument('--max-images', type=int,
                       help='Nombre maximum d images a traiter (defaut: toutes)')
    parser.add_argument('--image-list', type=str, nargs='+',
                       help='Liste specifique d images a analyser (chemins relatifs ou absolus)')
    parser.add_argument('--error-min', type=float,
                       help='Valeur minimale pour l echelle de couleur (pixels)')
    parser.add_argument('--error-max', type=float,
                       help='Valeur maximale pour l echelle de couleur (pixels)')
    parser.add_argument('--exaggeration', type=float, default=10,
                       help='Facteur d exageration des vecteurs d erreur (defaut: 10)')
    
    args = parser.parse_args()
    
    visualize_reprojection_errors(
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
        args.error_max,
        args.exaggeration
    )


if __name__ == "__main__":
    main()

