#!/usr/bin/env python3
"""
Calibration de caméra avec planches ChArUco
Modèle radial standard (k1, k2, k3, p1, p2)
"""

import cv2
import numpy as np
import json
import argparse
import xml.etree.ElementTree as ET
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


def calibrate_camera(images_folder, square_size_cm, squares_x=11, squares_y=8, marker_ratio=0.7):
    """Calibre une caméra à partir d'images de planches ChArUco"""
    
    print(f"=== CALIBRATION DE CAMERA ===")
    print(f"Dossier: {images_folder}")
    print(f"Taille carrés: {square_size_cm} cm")
    print(f"Configuration: {squares_x}x{squares_y}")
    print(f"Ratio marqueur: {marker_ratio}")
    print("=" * 40)
    
    # Créer la planche ChArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_size_cm, 
        square_size_cm * marker_ratio, 
        aruco_dict
    )
    
    # Trouver toutes les images
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
    
    # Dédupliquer les images (par chemin absolu pour éviter les doublons)
    seen_paths = set()
    unique_image_files = []
    for img_path in image_files:
        abs_path = img_path.resolve()  # Chemin absolu normalisé
        if abs_path not in seen_paths:
            seen_paths.add(abs_path)
            unique_image_files.append(img_path)
    
    image_files = unique_image_files
    
    print(f"Images trouvées: {len(image_files)} (après déduplication)")
    
    # Détecter les coins dans toutes les images
    all_corners = []
    all_ids = []
    valid_images = []
    
    for image_path in image_files:
        corners, ids = detect_charuco_corners(image_path, board)
        if corners is not None and ids is not None and len(ids) >= 4:
            all_corners.append(corners)
            all_ids.append(ids)
            valid_images.append(str(image_path))
            print(f"✓ {image_path.name}: {len(ids)} coins")
        else:
            print(f"✗ {image_path.name}: pas de détection")
    
    print(f"\nImages valides: {len(valid_images)}/{len(image_files)}")
    
    if len(valid_images) < 5:
        raise ValueError(f"Pas assez d'images valides ({len(valid_images)} < 5)")
    
    # Calibrer la caméra
    print("\nCalcul des paramètres...")
    first_image = cv2.imread(valid_images[0])
    image_size = (first_image.shape[1], first_image.shape[0])
    
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None
    )
    
    if not retval:
        raise RuntimeError("Échec de la calibration")
    
    print("✓ Calibration réussie!")
    print(f"Erreur de reprojection: {retval:.3f} pixels")
    
    # Extraire les paramètres
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"\nParamètres de la caméra:")
    print(f"  Focale: {fx:.1f} x {fy:.1f} pixels")
    print(f"  Point principal: ({cx:.1f}, {cy:.1f})")
    print(f"  Distorsion: k1={dist_coeffs[0][0]:.6f}, k2={dist_coeffs[0][1]:.6f}, k3={dist_coeffs[0][4]:.6f}")
    print(f"  Tangentiels: p1={dist_coeffs[0][2]:.6f}, p2={dist_coeffs[0][3]:.6f}")
    
    # Préparer les résultats
    # Sauvegarder les coins détectés pour éviter de les recalculer
    saved_corners = []
    saved_ids = []
    for corners, ids in zip(all_corners, all_ids):
        # Convertir en listes pour JSON
        if corners is not None:
            saved_corners.append(corners.tolist())
        else:
            saved_corners.append(None)
        if ids is not None:
            saved_ids.append(ids.tolist())
        else:
            saved_ids.append(None)
    
    results = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'image_size': image_size,
        'focal_length': [float(fx), float(fy)],
        'principal_point': [float(cx), float(cy)],
        'distortion_center': [float(cx), float(cy)],  # PP = CDist pour calibration standard
        'reprojection_error': float(retval),
        'valid_images': len(valid_images),
        'total_images': len(image_files),
        'rvecs': [rvec.tolist() for rvec in rvecs],
        'tvecs': [tvec.tolist() for tvec in tvecs],
        'valid_image_paths': valid_images,
        'detected_corners': saved_corners,  # Coins détectés sauvegardés
        'detected_ids': saved_ids  # IDs des coins sauvegardés
    }
    
    return results


def export_micmac(results, output_path):
    """Exporte au format MicMac (ModPhgrStd - Fraser avec coefficients tangentiels)"""
    camera_matrix = results['camera_matrix']
    dist_coeffs = results['distortion_coefficients'][0]
    image_size = results['image_size']
    
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]
    
    # Lire distortion_center si disponible (PP ≠ CDist), sinon utiliser PP
    if 'distortion_center' in results:
        cx_dist = results['distortion_center'][0]
        cy_dist = results['distortion_center'][1]
    else:
        # Fallback sur PP pour compatibilité avec anciennes calibrations
        cx_dist = cx
        cy_dist = cy
    
    # Conversion OpenCV → MicMac (normalisation par la focale)
    # ModPhgrStd utilise R3, R5, R7 (coefficients radiaux) + P1, P2 (tangentiels)
    r3 = dist_coeffs[0] / (fx * fx)  # k1 / f²
    r5 = dist_coeffs[1] / (fx * fx * fx * fx)  # k2 / f⁴
    r7 = dist_coeffs[4] / (fx * fx * fx * fx * fx * fx) if len(dist_coeffs) > 4 else 0.0  # k3 / f⁶
    
    # Coefficients tangentiels
    p1 = dist_coeffs[2] / fx if len(dist_coeffs) > 2 else 0.0  # p1 / f
    p2 = dist_coeffs[3] / fx if len(dist_coeffs) > 3 else 0.0  # p2 / f
    
    # Création du XML MicMac
    root = ET.Element("ExportAPERO")
    calib = ET.SubElement(root, "CalibrationInternConique")
    
    ET.SubElement(calib, "KnownConv").text = "eConvApero_DistM2C"
    ET.SubElement(calib, "PP").text = f"{cx} {cy}"
    ET.SubElement(calib, "F").text = str(fx)
    ET.SubElement(calib, "SzIm").text = f"{image_size[0]} {image_size[1]}"
    
    # Distorsion Fraser (ModPhgrStd)
    calib_dist = ET.SubElement(calib, "CalibDistortion")
    mod_phgr = ET.SubElement(calib_dist, "ModPhgrStd")
    
    radiale_part = ET.SubElement(mod_phgr, "RadialePart")
    ET.SubElement(radiale_part, "CDist").text = f"{cx_dist} {cy_dist}"  # Utilise CDist si PP ≠ CDist
    ET.SubElement(radiale_part, "CoeffDist").text = f"{r3:.10e}"
    ET.SubElement(radiale_part, "CoeffDist").text = f"{r5:.10e}"
    ET.SubElement(radiale_part, "CoeffDist").text = f"{r7:.10e}"
    
    # Coefficients tangentiels
    ET.SubElement(mod_phgr, "P1").text = f"{p1:.10e}"
    ET.SubElement(mod_phgr, "P2").text = f"{p2:.10e}"
    
    # Coefficients affines (zéro par défaut)
    ET.SubElement(mod_phgr, "b1").text = "0.0"
    ET.SubElement(mod_phgr, "b2").text = "0.0"
    
    # Sauvegarde
    tree = ET.ElementTree(root)
    ET.indent(tree, space="     ", level=0)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"✓ Format MicMac exporté: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibration de caméra ChArUco')
    parser.add_argument('images_folder', help='Dossier contenant les images')
    parser.add_argument('--square-size', type=float, required=True, 
                       help='Taille des carrés en cm')
    parser.add_argument('--squares-x', type=int, default=11, 
                       help='Nombre de carrés en X (défaut: 11)')
    parser.add_argument('--squares-y', type=int, default=8, 
                       help='Nombre de carrés en Y (défaut: 8)')
    parser.add_argument('--marker-ratio', type=float, default=0.7,
                       help='Ratio taille marqueur/carré (défaut: 0.7)')
    parser.add_argument('--output', default='calibration.json', 
                       help='Fichier de sortie JSON (défaut: calibration.json)')
    parser.add_argument('--export-micmac', action='store_true',
                       help='Exporter également au format MicMac XML')
    
    args = parser.parse_args()
    
    # Calibration
    results = calibrate_camera(
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio
    )
    
    # Sauvegarde JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {args.output}")
    
    # Export MicMac si demandé
    if args.export_micmac:
        micmac_path = args.output.replace('.json', '_micmac.xml')
        export_micmac(results, micmac_path)


if __name__ == "__main__":
    main()
