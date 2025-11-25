#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur simple de planches ChArUco
- Taille de carré personnalisable
- Marqueurs ArUco 2x plus petits que les carrés
- Export PDF avec bonnes dimensions et DPI
"""

import cv2
import numpy as np
from PIL import Image
import argparse
import os

def create_charuco_board(square_size_cm, squares_x=11, squares_y=8, output_name="charuco_board"):
    """
    Crée une planche ChArUco avec les paramètres spécifiés
    
    Args:
        square_size_cm: Taille d'un carré en cm
        squares_x: Nombre de carrés en largeur
        squares_y: Nombre de carrés en hauteur
        output_name: Nom de base des fichiers de sortie
    """
    
    # Calculer les tailles en mètres
    square_length = square_size_cm / 100.0  # Convertir cm en m
    marker_length = square_length * 0.7     # Marqueurs 70% de la taille des carrés
    
    print(f"=== GENERATION DE PLANCHE CHARUCO ===")
    print(f"Taille carre: {square_size_cm} cm")
    print(f"Taille marqueur: {marker_length*100:.1f} cm")
    print(f"Configuration: {squares_x}x{squares_y} carres")
    
    # Créer le dictionnaire ArUco
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Créer la planche ChArUco
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )
    
    # Calculer les dimensions en pixels pour 300 DPI
    dpi = 300
    pixels_per_cm = dpi / 2.54
    
    # Dimensions réelles en cm
    real_width_cm = squares_x * square_size_cm
    real_height_cm = squares_y * square_size_cm
    
    # Dimensions en pixels
    width_px = int(real_width_cm * pixels_per_cm)
    height_px = int(real_height_cm * pixels_per_cm)
    
    print(f"Dimensions reelles: {real_width_cm:.1f} x {real_height_cm:.1f} cm")
    print(f"Dimensions pixels: {width_px} x {height_px}")
    print(f"Resolution: {dpi} DPI")
    
    # Générer l'image de la planche
    board_image = board.generateImage((width_px, height_px), marginSize=0)
    
    # Convertir en PIL Image pour contrôler les métadonnées
    if len(board_image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(board_image)
    
    # Sauvegarder PNG avec métadonnées DPI
    png_filename = f"{output_name}.png"
    pil_image.save(png_filename, dpi=(dpi, dpi), optimize=True)
    print(f"+ PNG genere: {png_filename}")
    
    # Créer version avec marges pour PDF
    margin_cm = 1.0
    total_width_cm = real_width_cm + 2 * margin_cm
    total_height_cm = real_height_cm + 2 * margin_cm
    
    total_width_px = int(total_width_cm * pixels_per_cm)
    total_height_px = int(total_height_cm * pixels_per_cm)
    margin_px = int(margin_cm * pixels_per_cm)
    
    # Créer le fond blanc avec marges
    marged_image = Image.new('RGB', (total_width_px, total_height_px), 'white')
    marged_image.paste(pil_image, (margin_px, margin_px))
    
    # Sauvegarder PNG avec marges
    png_marged_filename = f"{output_name}_with_margins.png"
    marged_image.save(png_marged_filename, dpi=(dpi, dpi), optimize=True)
    print(f"+ PNG avec marges: {png_marged_filename}")
    
    # Créer PDF
    pdf_filename = f"{output_name}.pdf"
    marged_image.save(pdf_filename, format='PDF', resolution=300.0, quality=95)
    print(f"+ PDF genere: {pdf_filename}")
    
    print(f"\nDimensions finales:")
    print(f"  Planche seule: {real_width_cm:.1f} x {real_height_cm:.1f} cm")
    print(f"  Avec marges: {total_width_cm:.1f} x {total_height_cm:.1f} cm")
    
    return png_filename, png_marged_filename, pdf_filename

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Générateur de planches ChArUco')
    parser.add_argument('--square-size', type=float, default=2.0, 
                       help='Taille d\'un carré en cm (défaut: 2.0)')
    parser.add_argument('--squares-x', type=int, default=11,
                       help='Nombre de carrés en largeur (défaut: 11)')
    parser.add_argument('--squares-y', type=int, default=8,
                       help='Nombre de carrés en hauteur (défaut: 8)')
    parser.add_argument('--output', type=str, default='charuco_board',
                       help='Nom de base des fichiers de sortie (défaut: charuco_board)')
    
    args = parser.parse_args()
    
    print("GENERATEUR DE PLANCHE CHARUCO")
    print("=" * 40)
    
    try:
        png_file, png_marged_file, pdf_file = create_charuco_board(
            square_size_cm=args.square_size,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            output_name=args.output
        )
        
        print("\n" + "=" * 40)
        print("SUCCES: Planche ChArUco generee!")
        print(f"\nFichiers crees:")
        print(f"  - {png_file} (planche exacte)")
        print(f"  - {png_marged_file} (avec marges)")
        print(f"  - {pdf_file} (recommandé pour impression)")
        
        print(f"\nPour imprimer:")
        print(f"  1. Ouvrir {pdf_file}")
        print(f"  2. Imprimer en 100% (taille réelle)")
        print(f"  3. Vérifier: {args.squares_x * args.square_size:.1f} x {args.squares_y * args.square_size:.1f} cm")
        
    except Exception as e:
        print(f"ERREUR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
