import cv2
import time
import logging
import math
from ultralytics import YOLO
import requests
import torch

# ==================== CONFIGURACIÓN ==================== #
MODEL_PATH = "runs/detect/ppe_yolo11n_180ep/weights/best.pt"
RTSP_URL = "rtsp://admin:@jp2025.@192.168.1.204:554/Streaming/Channels/101"

CLASS_NAMES = {0: "Persona", 1: "Gafas", 2: "Casco", 3: "Celular"}
MAX_HEAD_DISTANCE = 220
TIME_BEFORE_ALERT = 1
ALERT_COOLDOWN = 60
AREA = "Planta principal Bavaria Monteria picking 1"

# --- Telegram ---
TELEGRAM_TOKEN = "7960172584:AAFot_eGtO5xk2fIKGtw0UW3bw5_1xqJRyE"
CHAT_ID = "8152029002"

# --- Logging ---
logging.basicConfig(filename="deteccion.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ==================== FUNCIONES ==================== #
def connect_camera(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logging.error(f"No se pudo conectar a la cámara: {url}")
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def safe_read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        logging.warning("Error leyendo cámara. Reconectando...")
        cap.release()
        time.sleep(5)
        cap = connect_camera(RTSP_URL)
        return False, None, cap
    return True, frame, cap

def is_helmet_associated(person_box, casco_coords, max_dist):
    x1, y1, x2, y2 = person_box
    head_center_x = (x1 + x2) // 2
    head_center_y = y1 + (y2 - y1) // 4
    for cx1, cy1, cx2, cy2 in casco_coords:
        casco_center_x = (cx1 + cx2) // 2
        casco_center_y = (cy1 + cy2) // 2
        distance = math.sqrt((head_center_x - casco_center_x)**2 + (head_center_y - casco_center_y)**2)

        if distance < max_dist:
            return True
    return False

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text})
        r.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Error enviando mensaje Telegram: {e}")
        return False

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo:
            r = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})
            r.raise_for_status()
            return True
    except Exception as e:
        logging.error(f"Error enviando foto Telegram: {e}")
        return False

def save_frame(frame, track_id, timestamp):
    filename = f"alert_{track_id}_{int(timestamp)}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Foto guardada: {filename}")  # Debug
    return filename

# ==================== BUCLE PRINCIPAL ==================== #
def main():
    # Cargar modelo y configurar dispositivo
    model = YOLO(MODEL_PATH)
    
    # Verificar y configurar GPU
    if torch.cuda.is_available():
        model.to('cuda')
        print("✅ Ejecutando en GPU")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ Ejecutando en CPU - GPU no disponible")
    
    print(f"Nombres de clases del modelo: {model.names}")
    print(f"Dispositivo actual: {model.device}")
    
    cap = connect_camera(RTSP_URL)
    if cap is None:
        return

    estado_personas = {}
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame, cap = safe_read_frame(cap)
        if not ret:
            if cap is None:
                logging.critical("Conexión con la cámara falló permanentemente. Saliendo.")
                break
            continue

        current_time = time.time()
        frame_count += 1
        
        # Hacer una COPIA del frame original para dibujar
        display_frame = frame.copy()
        
        # Configurar dispositivo para inference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = model.track(
            frame, 
            imgsz=640, 
            conf=0.4, 
            persist=True, 
            verbose=False,
            device=device
        )
        
        if not results or not results[0].boxes:
            cv2.imshow("Detección EPP", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            continue

        detections = results[0].boxes
        personas = [d for d in detections if CLASS_NAMES.get(int(d.cls)) == "Persona" and d.id is not None]
        cascos = [d for d in detections if CLASS_NAMES.get(int(d.cls)) == "Casco"]
        cascos_coords = [list(map(int, c.xyxy[0].tolist())) for c in cascos]

        detected_ids = set()
        alert_frame = None  # Frame para guardar en alerta

        for persona in personas:
            x1, y1, x2, y2 = map(int, persona.xyxy[0].tolist())
            track_id = int(persona.id.item())
            detected_ids.add(track_id)

            has_helmet = is_helmet_associated((x1, y1, x2, y2), cascos_coords, MAX_HEAD_DISTANCE)

            if track_id not in estado_personas:
                estado_personas[track_id] = {"estado": "con_epp", "tiempo_sin_epp": current_time, "ultimo_alerta": 0}

            persona_data = estado_personas[track_id]

            if has_helmet:
                # DIBUJAR en el frame de display
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID {track_id} - CON CASCO", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if persona_data["estado"] == "sin_epp":
                    persona_data["estado"] = "con_epp"
                    send_telegram_message(f"Persona {track_id} se puso el casco nuevamente.")
                    logging.info(f"Persona {track_id} recuperó el casco")
            else:
                if persona_data["estado"] == "con_epp":
                    persona_data["estado"] = "sin_epp"
                    persona_data["tiempo_sin_epp"] = current_time

                elapsed = current_time - persona_data["tiempo_sin_epp"]

                # DIBUJAR en el frame de display (SIEMPRE)
                color = (0, 0, 255)
                text = f"ID {track_id} - {'SIN CASCO' if elapsed >= TIME_BEFORE_ALERT else f'Sin casco ({int(TIME_BEFORE_ALERT - elapsed)}s)'}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Crear un frame ESPECIAL para la alerta
                if elapsed >= TIME_BEFORE_ALERT and (current_time - persona_data["ultimo_alerta"]) > ALERT_COOLDOWN:
                    # Crear una COPIA del frame de display con TODO dibujado
                    alert_frame = display_frame.copy()
                    
                    filename = save_frame(alert_frame, track_id, current_time)
                    caption = (
                        f"ALERTA EPP: Personal sin casco\n"
                        f"ID: {track_id}\n"
                        f"Tiempo sin casco: {int(elapsed)}s\n"
                        f"Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Área: {AREA}"
                    )
                    if send_telegram_photo(filename, caption):
                        persona_data["ultimo_alerta"] = current_time
                        logging.warning(f"Alerta enviada Telegram ID {track_id}")
                        print(f"🚨 Alerta enviada para ID {track_id}")

        # Dibujar cascos en el frame de display
        for (cx1, cy1, cx2, cy2) in cascos_coords:
            cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
            cv2.putText(display_frame, "Casco", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Limpiar IDs que ya no están
        ids_to_remove = [tid for tid in estado_personas if tid not in detected_ids]
        for tid in ids_to_remove:
            if (current_time - estado_personas[tid]["ultimo_alerta"]) > 5:
                del estado_personas[tid]

        # Calcular y mostrar FPS
        fps = frame_count / (current_time - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Device: {device.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # DEBUG: Guardar frame de prueba ocasionalmente
        if frame_count % 100 == 0:  # Cada 100 frames
            cv2.imwrite("debug_frame.jpg", display_frame)
            print("📸 Debug frame guardado: debug_frame.jpg")

        cv2.imshow("Detección EPP", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()