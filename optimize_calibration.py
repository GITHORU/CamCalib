#!/usr/bin/env python3
"""
Optimisation avancée de calibration avec PP et CDist séparés
Ré-estimation de tous les paramètres pour haute précision métrologique
"""

import cv2
import numpy as np
import json
import argparse
from scipy.optimize import least_squares
from pathlib import Path


def load_calibration_results(json_path):
    """Charge les résultats de calibration OpenCV"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def load_charuco_data(images_folder, board, square_size_cm):
    """Charge toutes les données ChArUco détectées"""
    from camera_calibrator import detect_charuco_corners
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f'*{ext}'))
        image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))
    
    all_corners = []
    all_ids = []
    all_obj_points = []
    valid_images = []
    obj_points = board.getChessboardCorners()
    
    for image_path in image_files:
        corners, ids = detect_charuco_corners(str(image_path), board)
        if corners is not None and ids is not None:
            # Convertir en points 3D et 2D
            obj_pts = []
            img_pts = []
            valid_ids = []
            
            # Gérer les IDs (peuvent être un array numpy)
            if isinstance(ids, np.ndarray):
                ids_flat = ids.flatten()
            else:
                ids_flat = ids
            
            # Gérer les corners (peuvent être de différentes formes)
            if len(corners.shape) > 2:
                corners_flat = corners.reshape(-1, 2)
            else:
                corners_flat = corners
            
            for i in range(min(len(corners_flat), len(ids_flat))):
                if ids_flat[i] is not None:
                    try:
                        obj_idx = int(ids_flat[i])
                        if 0 <= obj_idx < len(obj_points):
                            obj_pts.append(obj_points[obj_idx])
                            img_pts.append(corners_flat[i])
                            valid_ids.append(obj_idx)
                    except (ValueError, IndexError, TypeError):
                        continue
            
            if len(obj_pts) >= 4:
                all_corners.append(np.array(img_pts, dtype=np.float32))
                all_ids.append(np.array(valid_ids, dtype=np.int32))
                all_obj_points.append(np.array(obj_pts, dtype=np.float32))
                valid_images.append(str(image_path))
    
    return all_corners, all_ids, all_obj_points, valid_images


def project_points_with_separate_cdist(obj_points, rvec, tvec, fx, fy, cx, cy, 
                                       cx_dist, cy_dist, k1, k2, k3, p1, p2):
    """
    Projette les points 3D en 2D avec PP et CDist séparés
    
    Args:
        obj_points: Points 3D (N, 3)
        rvec: Vecteur de rotation (3,)
        tvec: Vecteur de translation (3,)
        fx, fy: Focales
        cx, cy: Point principal (PP)
        cx_dist, cy_dist: Centre de distorsion (CDist)
        k1, k2, k3: Coefficients radiaux
        p1, p2: Coefficients tangentiels
    
    Returns:
        Points projetés (N, 2)
    """
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
    
    # Remettre dans le système image
    x_final = x_dist * fx + cx_dist
    y_final = y_dist * fy + cy_dist
    
    return np.column_stack([x_final, y_final])


def compute_reprojection_error(params, all_obj_points, all_img_points, image_size):
    """
    Calcule l'erreur de reprojection pour tous les points
    
    Args:
        params: Vecteur de paramètres [fx, fy, cx, cy, cx_dist, cy_dist, k1, k2, k3, p1, p2, ...poses...]
        all_obj_points: Liste de tableaux de points 3D pour chaque image
        all_img_points: Liste de tableaux de points 2D observés pour chaque image
        image_size: Taille de l'image (width, height)
    
    Returns:
        Vecteur d'erreurs (flattened)
    """
    # Extraire les paramètres intrinsèques
    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    cx_dist = params[4]
    cy_dist = params[5]
    k1 = params[6]
    k2 = params[7]
    k3 = params[8]
    p1 = params[9]
    p2 = params[10]
    
    # Nombre de paramètres par pose (6: 3 rotation + 3 translation)
    n_intrinsic = 11
    n_pose_params = 6
    
    errors = []
    
    for i, (obj_pts, img_pts) in enumerate(zip(all_obj_points, all_img_points)):
        # Extraire la pose pour cette image
        pose_idx = n_intrinsic + i * n_pose_params
        rvec = params[pose_idx:pose_idx+3]
        tvec = params[pose_idx+3:pose_idx+6]
        
        # Projeter les points
        projected = project_points_with_separate_cdist(
            obj_pts, rvec, tvec, fx, fy, cx, cy, cx_dist, cy_dist, k1, k2, k3, p1, p2
        )
        
        # Calculer l'erreur
        error = (projected - img_pts).ravel()
        errors.extend(error)
    
    return np.array(errors)


def optimize_calibration(calibration_json, images_folder, square_size_cm, 
                       squares_x=11, squares_y=8, marker_ratio=0.7):
    """
    Optimise la calibration avec PP et CDist séparés
    
    Args:
        calibration_json: Chemin vers le fichier JSON de calibration OpenCV
        images_folder: Dossier contenant les images
        square_size_cm: Taille des carrés en cm
        squares_x, squares_y: Dimensions de la planche
        marker_ratio: Ratio marqueur/carré
    """
    print("=== OPTIMISATION CALIBRATION AVANCEE ===")
    print(f"Calibration initiale: {calibration_json}")
    print(f"Dossier images: {images_folder}")
    print("=" * 50)
    
    # Charger la calibration initiale
    print("\n1. Chargement de la calibration initiale...")
    initial_results = load_calibration_results(calibration_json)
    
    camera_matrix = np.array(initial_results['camera_matrix'])
    # Les coefficients de distorsion peuvent être imbriqués: [[[k1, k2, p1, p2, k3]]]
    dist_coeffs_raw = initial_results['distortion_coefficients']
    if isinstance(dist_coeffs_raw[0], list):
        if isinstance(dist_coeffs_raw[0][0], list):
            dist_coeffs = np.array(dist_coeffs_raw[0][0])
        else:
            dist_coeffs = np.array(dist_coeffs_raw[0])
    else:
        dist_coeffs = np.array(dist_coeffs_raw)
    
    image_size = initial_results['image_size']
    
    fx_init = camera_matrix[0, 0]
    fy_init = camera_matrix[1, 1]
    cx_init = camera_matrix[0, 2]
    cy_init = camera_matrix[1, 2]
    
    k1_init = dist_coeffs[0]
    k2_init = dist_coeffs[1] if len(dist_coeffs) > 1 else 0.0
    k3_init = dist_coeffs[4] if len(dist_coeffs) > 4 else 0.0
    p1_init = dist_coeffs[2] if len(dist_coeffs) > 2 else 0.0
    p2_init = dist_coeffs[3] if len(dist_coeffs) > 3 else 0.0
    
    print(f"  Focale initiale: {fx_init:.2f} x {fy_init:.2f}")
    print(f"  PP initial: ({cx_init:.2f}, {cy_init:.2f})")
    print(f"  Erreur initiale: {initial_results['reprojection_error']:.4f} pixels")
    
    # Créer la planche ChArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size_cm, 
                                   square_size_cm * marker_ratio, aruco_dict)
    
    # Charger les données ChArUco
    print("\n2. Chargement des données ChArUco...")
    all_corners, all_ids, all_obj_points, valid_images = load_charuco_data(
        images_folder, board, square_size_cm
    )
    print(f"  Images valides: {len(valid_images)}")
    
    if len(valid_images) < 5:
        raise ValueError(f"Pas assez d'images valides ({len(valid_images)} < 5)")
    
    # Recalibrer pour obtenir les poses initiales
    print("\n3. Recalibration pour obtenir les poses initiales...")
    # Convertir les corners au format attendu par OpenCV (liste de tableaux 2D)
    corners_for_cv = []
    ids_for_cv = []
    for corners, ids in zip(all_corners, all_ids):
        # Reshape pour avoir la bonne forme (N, 1, 2)
        corners_reshaped = corners.reshape(-1, 1, 2).astype(np.float32)
        corners_for_cv.append(corners_reshaped)
        ids_for_cv.append(ids.reshape(-1, 1).astype(np.int32))
    
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        corners_for_cv, ids_for_cv, board, image_size, None, None, None, None, flags=0
    )
    
    # Construire le vecteur de paramètres initial
    print("\n4. Initialisation des paramètres...")
    n_images = len(all_obj_points)
    
    # Paramètres intrinsèques: [fx, fy, cx, cy, cx_dist, cy_dist, k1, k2, k3, p1, p2]
    # CDist initialisé à PP
    params_init = np.array([
        fx_init, fy_init,           # Focales
        cx_init, cy_init,           # PP
        cx_init, cy_init,           # CDist (initialisé à PP)
        k1_init, k2_init, k3_init,  # Radiaux
        p1_init, p2_init           # Tangentiels
    ])
    
    # Ajouter les poses (rvecs et tvecs)
    for rvec, tvec in zip(rvecs, tvecs):
        params_init = np.append(params_init, rvec.flatten())
        params_init = np.append(params_init, tvec.flatten())
    
    print(f"  Nombre total de paramètres: {len(params_init)}")
    print(f"  - Intrinsèques: 11")
    print(f"  - Extrinsèques: {6 * n_images} ({n_images} images)")
    
    # Définir les bornes pour l'optimisation
    print("\n5. Définition des bornes...")
    # Bornes pour les paramètres intrinsèques (larges pour permettre l'optimisation)
    bounds_lower = [
        fx_init * 0.3, fy_init * 0.3,  # Focales: 30% à 300%
        image_size[0] * 0.1, image_size[1] * 0.1,  # PP: 10% à 90% de l'image
        image_size[0] * 0.1, image_size[1] * 0.1,  # CDist: 10% à 90% de l'image
        -10.0, -10.0, -10.0,  # Radiaux: bornes très larges
        -1.0, -1.0            # Tangentiels: bornes larges
    ]
    
    bounds_upper = [
        fx_init * 3.0, fy_init * 3.0,
        image_size[0] * 0.9, image_size[1] * 0.9,
        image_size[0] * 0.9, image_size[1] * 0.9,
        10.0, 10.0, 10.0,
        1.0, 1.0
    ]
    
    # Bornes pour les poses (très larges)
    for _ in range(n_images):
        bounds_lower.extend([-np.pi, -np.pi, -np.pi, -10, -10, -10])
        bounds_upper.extend([np.pi, np.pi, np.pi, 10, 10, 10])
    
    bounds = (bounds_lower, bounds_upper)
    
    # Optimisation
    print("\n6. Optimisation non-linéaire (cela peut prendre du temps)...")
    print("   Méthode: Trust Region Reflective (TRF)")
    print("   Cela peut prendre plusieurs minutes selon le nombre d'images...")
    
    result = least_squares(
        compute_reprojection_error,
        params_init,
        args=(all_obj_points, all_corners, image_size),
        method='trf',  # Trust Region Reflective (supporte les bornes)
        bounds=bounds,
        verbose=2,
        max_nfev=1000,  # Maximum d'évaluations de fonction
        ftol=1e-6,  # Tolérance sur la fonction
        xtol=1e-8   # Tolérance sur les paramètres
    )
    
    if not result.success:
        print(f"\n⚠️  Optimisation non convergée: {result.message}")
    else:
        print(f"\n✅ Optimisation convergée!")
    
    # Extraire les résultats
    params_opt = result.x
    
    fx_opt = params_opt[0]
    fy_opt = params_opt[1]
    cx_opt = params_opt[2]
    cy_opt = params_opt[3]
    cx_dist_opt = params_opt[4]
    cy_dist_opt = params_opt[5]
    k1_opt = params_opt[6]
    k2_opt = params_opt[7]
    k3_opt = params_opt[8]
    p1_opt = params_opt[9]
    p2_opt = params_opt[10]
    
    # Calculer l'erreur finale
    errors = compute_reprojection_error(params_opt, all_obj_points, all_corners, image_size)
    mean_error = np.sqrt(np.mean(errors**2))
    
    print("\n=== RESULTATS OPTIMISATION ===")
    print(f"Erreur de reprojection: {mean_error:.4f} pixels")
    print(f"  (Initiale: {initial_results['reprojection_error']:.4f} pixels)")
    improvement = ((initial_results['reprojection_error'] - mean_error) / 
                   initial_results['reprojection_error'] * 100)
    print(f"  Amélioration: {improvement:.2f}%")
    
    print(f"\nParamètres intrinsèques:")
    print(f"  Focale: {fx_opt:.2f} x {fy_opt:.2f}")
    print(f"  PP: ({cx_opt:.2f}, {cy_opt:.2f})")
    print(f"  CDist: ({cx_dist_opt:.2f}, {cy_dist_opt:.2f})")
    dist_diff = np.sqrt((cx_opt - cx_dist_opt)**2 + (cy_opt - cy_dist_opt)**2)
    print(f"  Différence PP-CDist: {dist_diff:.2f} pixels")
    
    print(f"\nCoefficients de distorsion:")
    print(f"  Radiaux: k1={k1_opt:.6e}, k2={k2_opt:.6e}, k3={k3_opt:.6e}")
    print(f"  Tangentiels: p1={p1_opt:.6e}, p2={p2_opt:.6e}")
    
    # Préparer les résultats
    optimized_results = {
        'success': result.success,
        'optimization_message': result.message,
        'camera_matrix': [[fx_opt, 0, cx_opt], [0, fy_opt, cy_opt], [0, 0, 1]],
        'distortion_coefficients': [[k1_opt, k2_opt, p1_opt, p2_opt, k3_opt]],
        'image_size': image_size,
        'focal_length': [fx_opt, fy_opt],
        'principal_point': [cx_opt, cy_opt],
        'distortion_center': [cx_dist_opt, cy_dist_opt],
        'pp_cdist_difference': dist_diff,
        'reprojection_error': mean_error,
        'initial_reprojection_error': initial_results['reprojection_error'],
        'improvement_percent': improvement,
        'valid_images': len(valid_images),
        'calibration_model': 'optimized_separate_pp_cdist'
    }
    
    return optimized_results


def main():
    parser = argparse.ArgumentParser(
        description='Optimisation avancée de calibration avec PP et CDist séparés'
    )
    parser.add_argument('calibration_json', help='Fichier JSON de calibration OpenCV initiale')
    parser.add_argument('images_folder', help='Dossier contenant les images')
    parser.add_argument('--square-size', type=float, default=2.0, help='Taille des carrés en cm')
    parser.add_argument('--squares-x', type=int, default=11, help='Nombre de carrés en X')
    parser.add_argument('--squares-y', type=int, default=8, help='Nombre de carrés en Y')
    parser.add_argument('--marker-ratio', type=float, default=0.7, help='Ratio marqueur/carré')
    parser.add_argument('--output', default='calibration_optimized.json', 
                       help='Fichier de sortie JSON')
    
    args = parser.parse_args()
    
    # Optimisation
    results = optimize_calibration(
        args.calibration_json,
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio
    )
    
    if not results['success']:
        print(f"\n❌ Échec de l'optimisation: {results['optimization_message']}")
        return
    
    # Sauvegarde
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Résultats sauvegardés: {args.output}")
    
    # Export MicMac si possible
    try:
        from camera_calibrator import export_micmac
        base_name = args.output.replace('.json', '')
        export_micmac(results, f"{base_name}_micmac.xml")
        print(f"✅ Format MicMac exporté: {base_name}_micmac.xml")
    except Exception as e:
        print(f"⚠️  Export MicMac échoué: {e}")


if __name__ == "__main__":
    main()

