#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur ultra-simple de planches ChArUco
"""

import cv2
import numpy as np
from PIL import Image

def make_board(size_cm, output_name=None):
    """Crée une planche ChArUco simple"""
    
    # Configuration
    squares_x = 11
    squares_y = 8
    square_length = size_cm / 100.0
    marker_length = square_length * 0.7
    
    print(f"Creation planche: {size_cm}cm carres")
    print(f"Marqueurs: {marker_length*100:.1f}cm")
    
    # Créer la planche
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )
    
    # Dimensions
    dpi = 300
    real_w = squares_x * size_cm
    real_h = squares_y * size_cm
    w_px = int(real_w * dpi / 2.54)
    h_px = int(real_h * dpi / 2.54)
    
    # Générer
    board_image = board.generateImage((w_px, h_px), marginSize=0)
    
    # Convertir
    if len(board_image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(board_image)
    
    # Nom de fichier
    if output_name:
        filename = f"{output_name}.pdf"
    else:
        filename = f"board_{size_cm}cm.pdf"
    
    # Avec marges pour PDF
    margin = 1.0
    total_w = real_w + 2 * margin
    total_h = real_h + 2 * margin
    total_w_px = int(total_w * dpi / 2.54)
    total_h_px = int(total_h * dpi / 2.54)
    margin_px = int(margin * dpi / 2.54)
    
    pdf_image = Image.new('RGB', (total_w_px, total_h_px), 'white')
    pdf_image.paste(pil_image, (margin_px, margin_px))
    pdf_image.save(filename, format='PDF', resolution=300.0)
    
    print(f"Fichier cree: {filename}")
    print(f"Dimensions: {total_w:.1f} x {total_h:.1f} cm")
    
    return filename

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_board.py <taille_cm> [nom_sortie]")
        print("Exemples:")
        print("  python simple_board.py 2                    # 2cm carrés")
        print("  python simple_board.py 1.5                  # 1.5cm carrés")
        print("  python simple_board.py 3 ma_planche         # 3cm carrés, nom personnalisé")
        print("  python simple_board.py 2 calibration_board  # 2cm carrés, nom personnalisé")
        sys.exit(1)
    
    try:
        size = float(sys.argv[1])
        output_name = sys.argv[2] if len(sys.argv) > 2 else None
        make_board(size, output_name)
    except:
        print("Erreur: Taille invalide")
        sys.exit(1)
