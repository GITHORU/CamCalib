# Guide de Calibration de Caméra

## 🎯 **Vue d'ensemble**

Ce guide vous explique comment utiliser le script de calibration pour calculer les paramètres de votre caméra à partir d'images de planches ChArUco.

## 📋 **Prérequis**

1. **Planche ChArUco imprimée** (générée avec nos scripts)
2. **Images de calibration** (5 minimum, 10+ recommandé)
3. **Caméra à calibrer**

## 📸 **Prise d'images de calibration**

### **Conseils pour de bonnes images :**

1. **Éclairage uniforme** : Évitez les ombres et reflets
2. **Angles variés** : Prenez des images sous différents angles
3. **Distances variées** : Proche, moyen, éloigné
4. **Planche complète** : La planche doit être entièrement visible
5. **Stabilité** : Évitez le flou de bougé

### **Positions recommandées :**
- Planche au centre de l'image
- Planche dans les coins
- Planche inclinée (30-45°)
- Planche de biais
- Différentes distances

## 🚀 **Utilisation du script**

### **Commande de base :**
```bash
python camera_calibrator.py mes_images
```

### **Avec paramètres personnalisés :**
```bash
python camera_calibrator.py mes_images --square-size 2.4 --squares-x 11 --squares-y 8 --output ma_calibration.json
```

### **Paramètres disponibles :**
- `--square-size` : Taille des carrés en cm (défaut: 2.0)
- `--squares-x` : Nombre de carrés en largeur (défaut: 11)
- `--squares-y` : Nombre de carrés en hauteur (défaut: 8)
- `--marker-ratio` : Ratio marqueur/carré (défaut: 0.7)
- `--min-images` : Nombre minimum d'images valides (défaut: 5)
- `--output` : Fichier de sortie (défaut: camera_calibration.json)

## 📁 **Structure des dossiers**

```
mon_projet/
├── calibration_images/          # Dossier avec vos images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── camera_calibrator.py         # Script de calibration
└── camera_calibration.json      # Résultats (généré)
```

## 📊 **Résultats de calibration**

Le script génère un fichier JSON contenant :

```json
{
  "success": true,
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "image_size": [width, height],
  "focal_length": [fx, fy],
  "principal_point": [cx, cy],
  "field_of_view": [fov_x, fov_y],
  "reprojection_error": 0.123,
  "valid_images": 8,
  "total_images": 10
}
```

### **Paramètres importants :**
- **`focal_length`** : Distance focale en pixels
- **`principal_point`** : Point principal (centre optique)
- **`distortion_coefficients`** : Coefficients de correction de distorsion
- **`reprojection_error`** : Erreur de reprojection (plus bas = mieux)

## ✅ **Critères de qualité**

### **Erreur de reprojection :**
- **< 0.5 pixels** : Excellente calibration
- **0.5 - 1.0 pixels** : Bonne calibration
- **1.0 - 2.0 pixels** : Calibration acceptable
- **> 2.0 pixels** : Recalibrer avec plus d'images

### **Nombre d'images valides :**
- **Minimum** : 5 images
- **Recommandé** : 10-20 images
- **Optimal** : 20+ images

## 🔧 **Dépannage**

### **"Aucune image trouvée"**
- Vérifiez le chemin du dossier
- Vérifiez les extensions (.jpg, .png, .bmp, .tiff)

### **"Pas assez d'images valides"**
- Augmentez le nombre d'images
- Vérifiez la qualité des images
- Vérifiez les paramètres de la planche

### **"Pas de détection"**
- Vérifiez l'éclairage
- Vérifiez que la planche est complète
- Vérifiez les paramètres de la planche

### **Erreur de reprojection élevée**
- Prenez plus d'images
- Variez les angles et distances
- Vérifiez la stabilité de la caméra

## 📝 **Exemple complet**

```bash
# 1. Créer une planche
python simple_board.py 2.4 ma_planche

# 2. Imprimer la planche

# 3. Prendre des photos de calibration
# (sauvegarder dans dossier "calibration_images")

# 4. Calibrer la caméra
python camera_calibrator.py calibration_images --square-size 2.4 --output ma_camera.json

# 5. Vérifier les résultats
# (erreur < 1.0 pixel recommandé)
```

## 🎯 **Conseils avancés**

1. **Utilisez un trépied** pour la stabilité
2. **Éclairage LED** pour éviter le scintillement
3. **Planche rigide** pour éviter les déformations
4. **Images en RAW** si possible pour plus de précision
5. **Calibrez dans les conditions d'usage** (même éclairage, etc.)

---

**Bon calibrage !** 🎯


