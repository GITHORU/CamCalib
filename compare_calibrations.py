#!/usr/bin/env python3
"""
Script pour comparer deux calibrations et visualiser les différences
sous forme de champ de vecteurs.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def load_calibration(json_path):
    """Charge un fichier de calibration JSON."""
    with open(json_path, 'r') as f:
        calib = json.load(f)
    
    # Extraire les paramètres
    camera_matrix = np.array(calib['camera_matrix'], dtype=np.float64)
    dist_coeffs = np.array(calib['distortion_coefficients'], dtype=np.float64)
    
    # Gérer le format imbriqué des dist_coeffs
    if dist_coeffs.ndim > 1:
        dist_coeffs = dist_coeffs.flatten()
    
    # S'assurer qu'on a au moins 5 coefficients
    if len(dist_coeffs) < 5:
        dist_coeffs = np.pad(dist_coeffs, (0, 5 - len(dist_coeffs)), 'constant')
    
    image_size = tuple(calib['image_size'])
    
    return {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': dist_coeffs,
        'image_size': image_size,
        'focal_length': calib.get('focal_length', [camera_matrix[0,0], camera_matrix[1,1]]),
        'principal_point': calib.get('principal_point', [camera_matrix[0,2], camera_matrix[1,2]])
    }


def create_grid_3d_points(image_size, camera_matrix, grid_spacing=100, distance=100.0):
    """
    Crée une grille de points 3D dans un plan perpendiculaire à l'axe optique.
    
    Args:
        image_size: (width, height) de l'image
        camera_matrix: Matrice de caméra pour convertir pixels en coordonnées 3D
        grid_spacing: Espacement de la grille en pixels
        distance: Distance du plan à la caméra (en unités arbitraires)
    
    Returns:
        points_3d: Array (N, 3) de points 3D
        grid_points_2d: Array (N, 2) de points de la grille en pixels
    """
    width, height = image_size
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Créer une grille de points dans le plan image
    x_coords = np.arange(0, width, grid_spacing)
    y_coords = np.arange(0, height, grid_spacing)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Convertir en coordonnées 3D dans un plan à distance 'distance'
    # Le plan est perpendiculaire à l'axe optique (Z = distance)
    points_2d = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Convertir les coordonnées pixels en coordonnées 3D normalisées
    # x_norm = (u - cx) / fx, y_norm = (v - cy) / fy
    x_norm = (points_2d[:, 0] - cx) / fx
    y_norm = (points_2d[:, 1] - cy) / fy
    
    # Points 3D dans un plan à distance 'distance' (Z = distance)
    points_3d = np.zeros((len(points_2d), 3))
    points_3d[:, 0] = x_norm * distance  # X = x_norm * Z
    points_3d[:, 1] = y_norm * distance  # Y = y_norm * Z
    points_3d[:, 2] = distance            # Z = distance
    
    return points_3d, points_2d


def project_points(points_3d, camera_matrix, dist_coeffs):
    """
    Projette des points 3D sur le plan image avec distorsion.
    
    Returns:
        points_2d: Array (N, 2) de points projetés
    """
    # Rotation et translation nulles (points déjà dans le repère caméra)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    
    # Projection avec distorsion
    points_2d, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    
    return points_2d.reshape(-1, 2)


def compare_calibrations(calib1_path, calib2_path, grid_spacing=100, 
                        distance=100.0, exaggeration=1.0, output_path=None,
                        error_min=None, error_max=None):
    """
    Compare deux calibrations et génère un champ de vecteurs.
    
    Args:
        calib1_path: Chemin vers le premier fichier de calibration
        calib2_path: Chemin vers le second fichier de calibration
        grid_spacing: Espacement de la grille en pixels
        distance: Distance du plan 3D à la caméra
        exaggeration: Facteur d'exagération pour les vecteurs
        output_path: Chemin de sortie pour la figure (optionnel)
    """
    print("=== COMPARAISON DE CALIBRATIONS ===")
    print(f"Calibration 1: {calib1_path}")
    print(f"Calibration 2: {calib2_path}")
    
    # Charger les calibrations
    calib1 = load_calibration(calib1_path)
    calib2 = load_calibration(calib2_path)
    
    # Vérifier que les tailles d'image sont compatibles
    if calib1['image_size'] != calib2['image_size']:
        print(f"⚠️  Attention: Tailles d'image différentes!")
        print(f"   Calib 1: {calib1['image_size']}")
        print(f"   Calib 2: {calib2['image_size']}")
        # Utiliser la taille de la première calibration
        image_size = calib1['image_size']
    else:
        image_size = calib1['image_size']
    
    print(f"\nTaille d'image: {image_size}")
    print(f"Espacement de grille: {grid_spacing} pixels")
    print(f"Distance du plan: {distance}")
    
    # Créer la grille de points 3D (utiliser la première calibration comme référence)
    points_3d, grid_points_2d = create_grid_3d_points(
        image_size, calib1['camera_matrix'], grid_spacing, distance
    )
    print(f"Nombre de points dans la grille: {len(points_3d)}")
    
    # Projeter avec les deux calibrations
    print("\nProjection avec calibration 1...")
    proj1 = project_points(points_3d, calib1['camera_matrix'], 
                           calib1['distortion_coefficients'])
    
    print("Projection avec calibration 2...")
    proj2 = project_points(points_3d, calib2['camera_matrix'], 
                           calib2['distortion_coefficients'])
    
    # Calculer les différences (vecteurs)
    diff_vectors = proj2 - proj1
    diff_magnitudes = np.linalg.norm(diff_vectors, axis=1)
    
    print(f"\n=== STATISTIQUES DES DIFFÉRENCES ===")
    print(f"Différence moyenne: {np.mean(diff_magnitudes):.4f} pixels")
    print(f"Différence médiane: {np.median(diff_magnitudes):.4f} pixels")
    print(f"Différence max: {np.max(diff_magnitudes):.4f} pixels")
    print(f"Différence std: {np.std(diff_magnitudes):.4f} pixels")
    
    # Visualisation
    print("\nGénération de la visualisation...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filtrer les points qui sont dans l'image
    mask = ((proj1[:, 0] >= 0) & (proj1[:, 0] < image_size[0]) &
            (proj1[:, 1] >= 0) & (proj1[:, 1] < image_size[1]) &
            (proj2[:, 0] >= 0) & (proj2[:, 0] < image_size[0]) &
            (proj2[:, 1] >= 0) & (proj2[:, 1] < image_size[1]))
    
    proj1_filtered = proj1[mask]
    diff_vectors_filtered = diff_vectors[mask] * exaggeration
    diff_magnitudes_filtered = diff_magnitudes[mask]
    
    # Debug: afficher les magnitudes avant et après exagération
    if len(diff_vectors[mask]) > 0:
        max_before = np.max(np.linalg.norm(diff_vectors[mask], axis=1))
        max_after = np.max(np.linalg.norm(diff_vectors_filtered, axis=1))
        print(f"Magnitude max avant exagération: {max_before:.4f} pixels")
        print(f"Magnitude max après exagération ({exaggeration}x): {max_after:.4f} pixels")
    
    # Créer le champ de vecteurs
    X = proj1_filtered[:, 0]
    Y = proj1_filtered[:, 1]
    U = diff_vectors_filtered[:, 0]
    V = diff_vectors_filtered[:, 1]
    
    # Préparer les couleurs basées sur les magnitudes réelles (sans exagération)
    # Les couleurs doivent représenter les vraies différences en pixels
    cmap = plt.cm.get_cmap('RdYlGn_r')
    
    # Déterminer les limites min/max pour la colorbar
    if error_min is None:
        vmin = 0.0
    else:
        vmin = float(error_min)
    
    if error_max is None:
        vmax = float(np.max(diff_magnitudes_filtered)) if np.max(diff_magnitudes_filtered) > 0 else 1.0
    else:
        vmax = float(error_max)
    
    # S'assurer que vmax > vmin
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    # Normaliser les magnitudes pour les couleurs
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Afficher les limites utilisées
    print(f"\nLimites de l'échelle de couleur: min={vmin:.4f} px, max={vmax:.4f} px")
    
    # Dessiner les flèches avec annotate pour avoir un contrôle exact sur la longueur
    # Avec annotate, on peut garantir que 1 pixel de différence = 1 pixel de flèche (avec ex=1)
    for i in range(len(X)):
        if diff_magnitudes_filtered[i] > 0:  # Ne dessiner que les vecteurs non nuls
            # Couleur basée sur la magnitude réelle (sans exagération)
            # Clipper la valeur dans la plage [vmin, vmax] pour la normalisation
            magnitude_clipped = np.clip(diff_magnitudes_filtered[i], vmin, vmax)
            color = cmap(norm(magnitude_clipped))
            
            # Point d'arrivée = point de départ + vecteur avec exagération
            # Les vecteurs sont déjà multipliés par exaggeration dans diff_vectors_filtered
            x_end = X[i] + diff_vectors_filtered[i, 0]
            y_end = Y[i] + diff_vectors_filtered[i, 1]
            
            # Dessiner la flèche avec annotate pour correspondance exacte pixel à pixel
            ax.annotate('', xy=(x_end, y_end), xytext=(X[i], Y[i]),
                       arrowprops=dict(arrowstyle='->', color=color, 
                                     alpha=0.7, lw=1.5))
    
    # Créer un ScalarMappable pour la colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Tableau vide, on utilise juste pour la colorbar
    
    # Ajouter une barre de couleur avec les vraies valeurs en pixels
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Magnitude de la différence (pixels)', rotation=270, labelpad=20)
    
    # Configuration de l'axe
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)  # Inverser Y pour avoir l'origine en haut à gauche
    ax.set_aspect('equal')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Champ de vecteurs de différence entre deux calibrations\n'
                 f'Facteur d\'exagération: {exaggeration}x', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Ajouter des informations dans un coin
    info_text = (f"Calib 1: {Path(calib1_path).name}\n"
                 f"Calib 2: {Path(calib2_path).name}\n"
                 f"Diff. moyenne: {np.mean(diff_magnitudes_filtered):.3f} px\n"
                 f"Diff. max: {np.max(diff_magnitudes_filtered):.3f} px")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure sauvegardée: {output_path}")
    else:
        plt.show()
    
    return {
        'mean_difference': float(np.mean(diff_magnitudes)),
        'median_difference': float(np.median(diff_magnitudes)),
        'max_difference': float(np.max(diff_magnitudes)),
        'std_difference': float(np.std(diff_magnitudes)),
        'num_points': len(points_3d)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare deux calibrations et affiche un champ de vecteurs des différences'
    )
    parser.add_argument('calib1', help='Premier fichier de calibration JSON')
    parser.add_argument('calib2', help='Second fichier de calibration JSON')
    parser.add_argument('--grid-spacing', type=int, default=100,
                       help='Espacement de la grille en pixels (défaut: 100)')
    parser.add_argument('--distance', type=float, default=100.0,
                       help='Distance du plan 3D à la caméra (défaut: 100.0)')
    parser.add_argument('--exaggeration', type=float, default=1.0,
                       help='Facteur d\'exagération des vecteurs (défaut: 1.0)')
    parser.add_argument('--error-min', type=float, default=None,
                       help='Valeur minimale pour l\'échelle de couleur (défaut: 0)')
    parser.add_argument('--error-max', type=float, default=None,
                       help='Valeur maximale pour l\'échelle de couleur (défaut: max des différences)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Chemin de sortie pour la figure (défaut: affichage interactif)')
    
    args = parser.parse_args()
    
    # Vérifier que les fichiers existent
    if not Path(args.calib1).exists():
        print(f"❌ Erreur: Fichier non trouvé: {args.calib1}")
        return
    
    if not Path(args.calib2).exists():
        print(f"❌ Erreur: Fichier non trouvé: {args.calib2}")
        return
    
    # Comparer les calibrations
    results = compare_calibrations(
        args.calib1,
        args.calib2,
        grid_spacing=args.grid_spacing,
        distance=args.distance,
        exaggeration=args.exaggeration,
        output_path=args.output,
        error_min=args.error_min,
        error_max=args.error_max
    )
    
    print("\n=== COMPARAISON TERMINÉE ===")


if __name__ == "__main__":
    main()

