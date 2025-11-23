import cv2
import numpy as np
import open3d as o3d
import time
import glob
import os
import pickle

def calibrate_single_camera(images_path, pattern_size=(9, 6)):
    """
    Calibrar una sola cámara usando imágenes de tablero de ajedrez
    """
    # Preparar puntos del objeto 3D (0,0,0), (1,0,0), (2,0,0) ...., (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Arrays para almacenar puntos
    objpoints = []  # Puntos 3D en espacio real
    imgpoints = []  # Puntos 2D en imagen
    
    # Buscar imágenes
    images = glob.glob(images_path)
    
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Encontrar esquinas del tablero
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:            
            # Refinar la posición de las esquinas
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            # Dibujar y mostrar (opcional)
            cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
            cv2.imshow('Calibración', img)
            cv2.waitKey(500)  # Mostrar por 0.5 segundos
    
    cv2.destroyAllWindows()
    
    # Calibrar cámara
    print("Calculando parámetros de la cámara...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calcular error de reproyección
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': mean_error/len(objpoints)
    }

def contorno(frame, backSub):
    fg_mask = backSub.apply(frame, learningRate=0.7)  # Learning rate más bajo
    retval, mask_thresh = cv2.threshold(fg_mask, 120, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_DIAMOND, (5, 5))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 500
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# INICIALIZAR OPEN3D PRIMERO
print("Inicializando Open3D...")
vis1 = o3d.visualization.Visualizer()
vis1.create_window("Contornos 3D PC", width=800, height=600)
vis2 = o3d.visualization.Visualizer()
vis2.create_window("Contornos 3D Movil", width=800, height=600)

# Crear point cloud inicial con algunos puntos
pcd1 = o3d.geometry.PointCloud()
pcd2 = o3d.geometry.PointCloud()
initial_points = np.array([
    [0, 0, 0],
    [0, 0, 0.1],
    [0, 0.1, 0],
    [0.1, 0, 0]
], dtype=np.float32)
pcd1.points = o3d.utility.Vector3dVector(initial_points)
pcd1.paint_uniform_color([1, 0, 0])
pcd2.points = o3d.utility.Vector3dVector(initial_points)
pcd2.paint_uniform_color([1, 0, 0])

vis1.add_geometry(pcd1)
vis2.add_geometry(pcd2)

# INICIALIZAR CÁMARA DESPUÉS
print("Inicializando cámara...")
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
backSub1 = cv2.createBackgroundSubtractorMOG2(200, 16)
backSub2 = cv2.createBackgroundSubtractorMOG2(200, 16)

calib1 = calibrate_single_camera(PATH, (9, 6))
calib2 = calibrate_single_camera(PATH, (9, 6))

# Dar tiempo para que se abran las ventanas
time.sleep(2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error opening video file")
else:
    print("Presiona ESC para salir")
    
    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            image_size1 = (frame1.shape[1], frame1.shape[0])  # (ancho, alto)
            image_size2 = (frame2.shape[1], frame2.shape[0])  # (ancho, alto)
            calib_data = calibrate_from_captured(image_size)

            if not ret1 or not ret2:
                break
            
            # Obtener contornos
            contours1 = contorno(frame1, backSub1)
            contours2 = contorno(frame2, backSub2)
            
            # Dibujar en OpenCV
            frame_display1 = frame1.copy()
            frame_display2 = frame2.copy()
            cv2.drawContours(frame_display1, contours1, -1, (0, 255, 0), 2)
            cv2.imshow('Contornos 2D PC', frame_display1)
            cv2.drawContours(frame_display2, contours2, -1, (0, 255, 0), 2)
            cv2.imshow('Contornos 2D Movil', frame_display2)
            
            # Convertir contornos a puntos 3D
            points_3d1 = []
            if contours1:
                for contour in contours1:
                    # Convertir de (n, 1, 2) a (n, 2)
                    points_2d = contour.reshape(-1, 2)
                    
                    # Normalizar y convertir a 3D
                    for x, y in points_2d:
                        # Normalizar a [0, 1]
                        x_norm = x / (2 * frame1.shape[1])
                        y_norm = y / (2 * frame1.shape[0])
                        
                        # Crear coordenada Z interesante
                        z = 0
                        points_3d1.append([x_norm, -y_norm, z])
            
            points_3d2 = []
            if contours2:
                for contour in contours2:
                    # Convertir de (n, 1, 2) a (n, 2)
                    points_2d = contour.reshape(-1, 2)
                    
                    # Normalizar y convertir a 3D
                    for x, y in points_2d:
                        # Normalizar a [0, 1]
                        x_norm = x / (2 * frame1.shape[1])
                        y_norm = y / (2 * frame1.shape[0])
                        
                        # Crear coordenada Z interesante
                        z = 0
                        points_3d2.append([x_norm, -y_norm, z])
            
            # Actualizar Open3D
            if points_3d1:
                points_array = np.array(points_3d1, dtype=np.float32)
                pcd1.points = o3d.utility.Vector3dVector(points_array)
                
                # Cambiar color para ver que se actualiza
                colors = np.random.rand(len(points_array), 3)
                pcd1.colors = o3d.utility.Vector3dVector(colors)
                print(f"Puntos 3D actualizados: {len(points_array)}")

            if points_3d2:
                points_array = np.array(points_3d2, dtype=np.float32)
                pcd2.points = o3d.utility.Vector3dVector(points_array)
                
                # Cambiar color para ver que se actualiza
                colors = np.random.rand(len(points_array), 3)
                pcd2.colors = o3d.utility.Vector3dVector(colors)
                print(f"Puntos 3D actualizados: {len(points_array)}")
            
            # Actualizar visualizador
            vis1.update_geometry(pcd1)
            vis1.poll_events()
            vis1.update_renderer()
            vis2.update_geometry(pcd2)
            vis2.poll_events()
            vis2.update_renderer()
            
            # Salir con ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        vis1.destroy_window()
        vis2.destroy_window()
        print("Programa terminado")