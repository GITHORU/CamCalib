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
import subprocess
import re
import shutil
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


def filter_border_corners(corners, ids, board, border_margin=0):
    """
    Filtre les coins du bord du damier ChArUco
    
    Args:
        corners: Coins détectés (N, 2) ou (N, 1, 2)
        ids: IDs des coins (N,) ou (N, 1)
        board: Objet CharucoBoard
        border_margin: Nombre de rangées/colonnes à exclure du bord (défaut: 0 = pas de filtre)
    
    Returns:
        corners_filtered, ids_filtered: Coins et IDs filtrés (même format que l'entrée)
    """
    if border_margin == 0:
        return corners, ids
    
    squares_x, squares_y = board.getChessboardSize()
    
    # Normaliser les formats d'entrée
    ids_flat = ids.flatten() if isinstance(ids, np.ndarray) else np.array(ids).flatten()
    
    # Créer le masque de validation
    valid_mask = []
    for corner_id in ids_flat:
        # ID = y * squares_x + x dans ChArUco
        y = corner_id // squares_x
        x = corner_id % squares_x
        
        # Garder seulement si pas sur le bord
        if (border_margin <= x < squares_x - border_margin and 
            border_margin <= y < squares_y - border_margin):
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    valid_mask = np.array(valid_mask)
    
    # Filtrer les IDs
    ids_filtered = ids[valid_mask]
    
    # Filtrer les corners (gérer différents formats)
    if len(corners.shape) == 2:
        # Format (N, 2)
        corners_filtered = corners[valid_mask]
    elif len(corners.shape) == 3:
        # Format (N, 1, 2)
        corners_filtered = corners[valid_mask]
    else:
        corners_filtered = corners[valid_mask]
    
    return corners_filtered, ids_filtered


def calibrate_camera(images_folder, square_size_cm, squares_x=11, squares_y=8, marker_ratio=0.7, border_margin=0):
    """Calibre une caméra à partir d'images de planches ChArUco
    
    Args:
        images_folder: Dossier contenant les images
        square_size_cm: Taille des carrés en cm
        squares_x: Nombre de carrés en X (défaut: 11)
        squares_y: Nombre de carrés en Y (défaut: 8)
        marker_ratio: Ratio taille marqueur/carré (défaut: 0.7)
        border_margin: Nombre de rangées/colonnes à exclure du bord (défaut: 0)
    """
    
    print(f"=== CALIBRATION DE CAMERA ===")
    print(f"Dossier: {images_folder}")
    print(f"Taille carrés: {square_size_cm} cm")
    print(f"Configuration: {squares_x}x{squares_y}")
    print(f"Ratio marqueur: {marker_ratio}")
    if border_margin > 0:
        print(f"Filtrage bord: {border_margin} rangée(s)/colonne(s) exclue(s)")
    print("=" * 40)
    
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
    total_corners_before = 0
    total_corners_after = 0
    
    for image_path in image_files:
        corners, ids = detect_charuco_corners(image_path, board)
        if corners is not None and ids is not None and len(ids) >= 4:
            n_before = len(ids)
            total_corners_before += n_before
            
            # Filtrer les coins du bord si demandé
            if border_margin > 0:
                corners, ids = filter_border_corners(corners, ids, board, border_margin)
            
            n_after = len(ids)
            total_corners_after += n_after
            
            # Vérifier qu'il reste assez de coins après filtrage
            if n_after >= 4:
                all_corners.append(corners)
                all_ids.append(ids)
                valid_images.append(str(image_path))
                if border_margin > 0:
                    print(f"✓ {image_path.name}: {n_before} → {n_after} coins (filtrage bord)")
                else:
                    print(f"✓ {image_path.name}: {n_after} coins")
            else:
                print(f"✗ {image_path.name}: pas assez de coins après filtrage ({n_after} < 4)")
        else:
            print(f"✗ {image_path.name}: pas de détection")
    
    if border_margin > 0:
        reduction_pct = 100 * (1 - total_corners_after / total_corners_before) if total_corners_before > 0 else 0
        print(f"\nFiltrage bord: {total_corners_before} → {total_corners_after} coins ({reduction_pct:.1f}% exclus)")
    
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


def prepare_micmac_figee(calibration_xml, images_folder, output_dir="Ori-Calib", focal_mm=None):
    """
    Prépare la structure Ori-Calib/ pour utiliser une calibration avec Tapas Figee
    
    Args:
        calibration_xml: Chemin vers le fichier XML de calibration MicMac
        images_folder: Dossier contenant les images pour lire les EXIF
        output_dir: Nom du répertoire de sortie (défaut: Ori-Calib)
        focal_mm: Focale en mm (optionnel, sera détectée depuis EXIF si non fournie)
    
    Returns:
        Chemin vers le fichier de calibration créé
    """
    calibration_xml = Path(calibration_xml)
    images_folder = Path(images_folder)
    output_dir = Path(output_dir)
    
    if not calibration_xml.exists():
        raise FileNotFoundError(f"Fichier de calibration introuvable: {calibration_xml}")
    
    # Détecter la focale depuis les EXIF si non fournie
    if focal_mm is None:
        print(f"\nLecture de la focale depuis les EXIF des images dans {images_folder}...")
        
        # Essayer avec exiftool d'abord (plus fiable)
        try:
            # Chercher une image dans le dossier
            image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(images_folder.glob(f'*{ext}')))
                image_files.extend(list(images_folder.glob(f'*{ext.upper()}')))
            
            if image_files:
                # Essayer avec exiftool sur la première image
                result = subprocess.run(
                    ['exiftool', '-FocalLength', '-q', '-q', str(image_files[0])],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    # Parser la sortie: "Focal Length : 28.0 mm"
                    match = re.search(r'Focal Length\s*:\s*([\d.]+)', result.stdout)
                    if match:
                        focal_mm = float(match.group(1))
                        print(f"  ✓ Focale détectée (exiftool): {focal_mm} mm")
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
            pass
        
        # Si exiftool n'a pas fonctionné, essayer avec PIL/Pillow
        if focal_mm is None:
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                if image_files:
                    img = Image.open(image_files[0])
                    exif = img.getexif()
                    
                    # Chercher la focale dans les EXIF
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'FocalLength':
                            focal_mm = float(value)
                            print(f"  ✓ Focale détectée (PIL): {focal_mm} mm")
                            break
            except (ImportError, AttributeError, ValueError, KeyError, IndexError):
                pass
        
        # Si toujours pas de focale, demander à l'utilisateur
        if focal_mm is None:
            print("  ⚠️  Impossible de détecter la focale automatiquement.")
            print("     Utilisez l'option --focal-mm pour la spécifier manuellement.")
            raise ValueError("Focale non détectée. Utilisez --focal-mm pour la spécifier manuellement.")
    
    # Calculer le nom du fichier (focale * 10)
    focal_code = int(round(focal_mm * 10))
    output_filename = f"AutoCal{focal_code}.xml"
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copier le fichier avec le bon nom
    output_path = output_dir / output_filename
    shutil.copy2(calibration_xml, output_path)
    
    print(f"\n✓ Structure MicMac créée:")
    print(f"  Répertoire: {output_dir}/")
    print(f"  Fichier: {output_filename}")
    print(f"  Focale utilisée: {focal_mm} mm (code: {focal_code})")
    print(f"\n  Utilisation avec Tapas:")
    print(f"  mm3d Tapas Figee \"pattern\" Out=Sortie InCal={output_dir}/")
    
    return str(output_path)


def prepare_existing_calibration_for_figee(calibration_xml, images_folder, output_dir="Ori-Calib", focal_mm=None):
    """
    Prépare la structure Ori-Calib/ pour un fichier de calibration existant
    
    Usage en ligne de commande:
        python camera_calibrator.py --prepare-existing calibration.xml images_folder
    """
    return prepare_micmac_figee(calibration_xml, images_folder, output_dir, focal_mm)


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
    parser.add_argument('--prepare-figee', action='store_true',
                       help='Préparer la structure Ori-Calib/ pour Tapas Figee (nécessite --export-micmac)')
    parser.add_argument('--prepare-existing', type=str, default=None,
                       help='Préparer une calibration XML existante pour Figee (chemin vers le XML)')
    parser.add_argument('--focal-mm', type=float, default=None,
                       help='Focale en mm (détectée depuis EXIF si non fournie)')
    parser.add_argument('--figee-dir', default='Ori-Calib',
                       help='Nom du répertoire pour Figee (défaut: Ori-Calib)')
    parser.add_argument('--border-margin', type=int, default=0,
                       help='Nombre de rangées/colonnes à exclure du bord (défaut: 0 = pas de filtre)')
    
    args = parser.parse_args()
    
    # Si on veut juste préparer une calibration existante
    if args.prepare_existing:
        prepare_existing_calibration_for_figee(
            args.prepare_existing,
            args.images_folder,
            output_dir=args.figee_dir,
            focal_mm=args.focal_mm
        )
        return
    
    # Calibration
    results = calibrate_camera(
        args.images_folder,
        args.square_size,
        args.squares_x,
        args.squares_y,
        args.marker_ratio,
        args.border_margin
    )
    
    # Sauvegarde JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {args.output}")
    
    # Export MicMac si demandé
    micmac_path = None
    if args.export_micmac:
        micmac_path = args.output.replace('.json', '_micmac.xml')
        export_micmac(results, micmac_path)
    
    # Préparer la structure Figee si demandé
    if args.prepare_figee:
        if not args.export_micmac:
            print("\n⚠️  --prepare-figee nécessite --export-micmac")
            print("   Utilisation de --export-micmac automatiquement...")
            if micmac_path is None:
                micmac_path = args.output.replace('.json', '_micmac.xml')
                export_micmac(results, micmac_path)
        
        if micmac_path:
            prepare_micmac_figee(
                micmac_path,
                args.images_folder,
                output_dir=args.figee_dir,
                focal_mm=args.focal_mm
            )


if __name__ == "__main__":
    main()
