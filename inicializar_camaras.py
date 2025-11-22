import cv2

def inicializar_camaras():
    # Inicializar cámaras
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    # Configurar cámaras
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verificar que las cámaras estén funcionando
    if not cap1.isOpened():
        print("Error: No se pudo abrir cámara 1")
        return
        
    if not cap2.isOpened():
        print("Error: No se pudo abrir cámara 2")
        cap1.release()
        return
    
    print("Cámaras iniciadas. Presiona ESC para salir")
    
    while True:
        # Leer frames de ambas cámaras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Mostrar frames si son válidos
        if ret1:
            cv2.imshow("Camara 1", frame1)
        
        if ret2:
            cv2.imshow("Camara 2", frame2)
        
        # Salir con ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    # Liberar recursos
    cap1.release()
    cap2.release()

    cv2.destroyAllWindows()