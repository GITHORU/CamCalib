#!/usr/bin/env python3
"""
Calibration de caméra avec planches ChArUco
Export MicMac Fraser compatible
"""

import cv2
import numpy as np
import json
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def detect_charuco_corners(image_path, board):
    """Détecte les coins ChArUco dans une image"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Détecter les marqueurs ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    
    # Interpoler les coins ChArUco
    result = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    num_corners, charuco_corners, charuco_ids = result[:3]
    
    if num_corners < 4:
        return None, None
    
    return charuco_corners, charuco_ids


def calibrate_camera(images_folder, square_size_cm, squares_x=11, squares_y=8, 
                    marker_ratio=0.7, min_images=5, min_corners_percent=30):
    """Calibre une caméra à partir d'images de planches ChArUco"""
    
    print(f"=== CALIBRATION DE CAMERA ===")
    print(f"Dossier images: {images_folder}")
    print(f"Taille carres: {square_size_cm} cm")
    print(f"Configuration: {squares_x}x{squares_y}")
    print(f"Ratio marqueur: {marker_ratio}")
    print(f"Seuil coins minimum: {min_corners_percent}%")
    print("=" * 40)
    
    # Créer la planche ChArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size_cm, square_size_cm * marker_ratio, aruco_dict)
    
    # Trouver toutes les images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f'*{ext}'))
        image_files.extend(Path(images_folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"Aucune image trouvée dans {images_folder}")
    
    print(f"Images trouvées: {len(image_files)}")
    
    # Détecter les coins dans toutes les images
    all_corners = []
    all_ids = []
    valid_images = []
    
    for i, image_path in enumerate(image_files):
        print(f"Traitement {i+1}/{len(image_files)}: {image_path.name}", end=" ")
        
        corners, ids = detect_charuco_corners(str(image_path), board)
        
        if corners is not None and ids is not None:
            all_corners.append(corners)
            all_ids.append(ids)
            valid_images.append(str(image_path))
            print("+ OK")
        else:
            print("- Pas de detection")
    
    print(f"\nImages valides: {len(valid_images)}/{len(image_files)}")
    
    if len(valid_images) < min_images:
        raise ValueError(f"Pas assez d'images valides ({len(valid_images)} < {min_images})")
    
    # Calculer le nombre théorique de coins
    theoretical_corners = (squares_x - 1) * (squares_y - 1)
    min_corners = int(theoretical_corners * min_corners_percent / 100)
    
    print(f"Images avec >= 4 coins: {len(all_corners)}/{len(image_files)}")
    print(f"Nombre théorique maximum de coins: {theoretical_corners}")
    print(f"Seuil minimum ({min_corners_percent}%): {min_corners} coins")
    
    # Filtrer les images avec suffisamment de coins
    filtered_corners = []
    filtered_ids = []
    filtered_images = []
    
    for i, corners in enumerate(all_corners):
        if len(corners) >= min_corners:
            filtered_corners.append(corners)
            filtered_ids.append(all_ids[i])
            filtered_images.append(valid_images[i])
    
    print(f"Images avec >= {min_corners} coins: {len(filtered_corners)}/{len(image_files)}")
    
    if len(filtered_corners) < min_images:
        raise ValueError(f"Pas assez d'images avec suffisamment de coins ({len(filtered_corners)} < {min_images})")
    
    # Calibrer la caméra
    print("\nCalcul des paramètres de calibration...")
    
    try:
        first_image = cv2.imread(filtered_images[0])
        image_size = (first_image.shape[1], first_image.shape[0])
        
        # Utiliser l'API ChArUco dédiée
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            filtered_corners, filtered_ids, board, image_size, None, None, None, None,
            flags=0  # Centre de distorsion libre
        )
        
        if not retval:
            raise RuntimeError("Échec de la calibration")
        
        print("+ Calibration reussie!")
        
        # L'erreur de reprojection est retournée par calibrateCameraCharuco
        mean_error = retval
        print(f"Erreur de reprojection: {mean_error:.3f} pixels")
        
        # Calculer les paramètres de la caméra
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Calculer le FOV
        fov_x = 2 * np.arctan(image_size[0] / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(image_size[1] / (2 * fy)) * 180 / np.pi
        
        print(f"Focal length: {fx:.1f} x {fy:.1f}")
        print(f"Principal point: {cx:.1f} x {cy:.1f}")
        print(f"FOV: {fov_x:.1f}deg x {fov_y:.1f}deg")
        
        # Préparer les résultats
        results = {
            'success': True,
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': [dist_coeffs.tolist()],
            'image_size': image_size,
            'focal_length': [fx, fy],
            'principal_point': [cx, cy],
            'field_of_view': [fov_x, fov_y],
            'reprojection_error': mean_error,
            'valid_images': len(filtered_corners),
            'total_images': len(image_files),
            'calibration_model': 'standard'
        }
        
        return results
        
    except Exception as e:
        print(f"ERREUR: {e}")
        return {'success': False, 'error': str(e)}


def export_micmac(results, output_path):
    """Exporte au format MicMac Fraser"""
    camera_matrix = results['camera_matrix']
    dist_coeffs = results['distortion_coefficients'][0][0]  # Double indexation
    image_size = results['image_size']
    
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]
    
    # Création du XML MicMac Fraser
    root = ET.Element("ExportAPERO")
    calib = ET.SubElement(root, "CalibrationInternConique")
    
    ET.SubElement(calib, "KnownConv").text = "eConvApero_DistM2C"
    ET.SubElement(calib, "PP").text = f"{cx} {cy}"
    ET.SubElement(calib, "F").text = str(fx)
    ET.SubElement(calib, "SzIm").text = f"{image_size[0]} {image_size[1]}"
    
    # Distorsion Fraser
    calib_dist = ET.SubElement(calib, "CalibDistortion")
    mod_phgr = ET.SubElement(calib_dist, "ModPhgrStd")
    
    radiale_part = ET.SubElement(mod_phgr, "RadialePart")
    ET.SubElement(radiale_part, "CDist").text = f"{cx} {cy}"
    
    # Conversion OpenCV → MicMac
    fx_val = results['focal_length'][0]
    
    r3_micmac = dist_coeffs[0] / (fx_val * fx_val)
    r5_micmac = dist_coeffs[1] / (fx_val * fx_val * fx_val * fx_val)
    r7_micmac = dist_coeffs[4] / (fx_val * fx_val * fx_val * fx_val * fx_val * fx_val) if len(dist_coeffs) > 4 else 0.0
    
    ET.SubElement(radiale_part, "CoeffDist").text = f"{r3_micmac:.10e}"
    ET.SubElement(radiale_part, "CoeffDist").text = f"{r5_micmac:.10e}"
    ET.SubElement(radiale_part, "CoeffDist").text = f"{r7_micmac:.10e}"
    
    # Les coefficients inverses sont calculés par MicMac lors de l'utilisation
    # Nous ne les incluons pas car nous ne connaissons pas l'algorithme exact
    
    # Coefficients tangentiels
    p1_micmac = dist_coeffs[2] / fx_val
    p2_micmac = dist_coeffs[3] / fx_val
    
    ET.SubElement(mod_phgr, "P1").text = f"{p1_micmac:.10e}"
    ET.SubElement(mod_phgr, "P2").text = f"{p2_micmac:.10e}"
    
    # Coefficients affines (zéro par défaut)
    ET.SubElement(mod_phgr, "b1").text = "0.0"
    ET.SubElement(mod_phgr, "b2").text = "0.0"
    
    # Sauvegarde
    tree = ET.ElementTree(root)
    ET.indent(tree, space="     ", level=0)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"✅ Format MicMac exporté : {output_path}")


def export_meshroom(results, output_path):
    """Exporte au format Meshroom"""
    camera_matrix = results['camera_matrix']
    dist_coeffs = results['distortion_coefficients'][0]
    image_size = results['image_size']
    
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]
    
    # Estimation des dimensions du capteur
    sensor_width = 36.0  # mm (format plein format)
    sensor_height = sensor_width * image_size[1] / image_size[0]
    
    meshroom_data = {
        "sensorWidth": sensor_width,
        "sensorHeight": sensor_height,
        "focalLength": fx,
        "principalPoint": [cx, cy],
        "distortionParams": dist_coeffs[:5].tolist(),
        "imageWidth": image_size[0],
        "imageHeight": image_size[1]
    }
    
    with open(output_path, 'w') as f:
        json.dump(meshroom_data, f, indent=2)
    
    print(f"✅ Format Meshroom exporté : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibration de caméra ChArUco')
    parser.add_argument('images_folder', help='Dossier contenant les images')
    parser.add_argument('--square-size', type=float, default=2.0, help='Taille des carrés en cm')
    parser.add_argument('--squares-x', type=int, default=11, help='Nombre de carrés en X')
    parser.add_argument('--squares-y', type=int, default=8, help='Nombre de carrés en Y')
    parser.add_argument('--marker-ratio', type=float, default=0.7, help='Ratio marqueur/carré')
    parser.add_argument('--min-corners-percent', type=int, default=30, help='Pourcentage minimum de coins détectés')
    parser.add_argument('--output', default='calibration.json', help='Fichier de sortie JSON')
    parser.add_argument('--export-formats', choices=['micmac', 'meshroom', 'both'], help='Formats d\'export')
    
    args = parser.parse_args()
    
    # Calibration
    results = calibrate_camera(
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio,
        min_corners_percent=args.min_corners_percent
    )
    
    if not results['success']:
        print(f"❌ Échec de la calibration: {results['error']}")
        return
    
    # Sauvegarde JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Résultats sauvegardés: {args.output}")
    
    # Export formats additionnels
    if args.export_formats:
        base_name = args.output.replace('.json', '')
        
        if args.export_formats in ['micmac', 'both']:
            export_micmac(results, f"{base_name}_micmac.xml")
        
        if args.export_formats in ['meshroom', 'both']:
            export_meshroom(results, f"{base_name}_meshroom.json")
        
        print(f"\n📁 Formats additionnels exportés:")
        if args.export_formats in ['micmac', 'both']:
            print(f"   - {base_name}_micmac.xml")
        if args.export_formats in ['meshroom', 'both']:
            print(f"   - {base_name}_meshroom.json")
    
    print(f"\n+ Calibration terminee avec succes!")
    print(f"Fichier: {args.output}")


if __name__ == "__main__":
    main()