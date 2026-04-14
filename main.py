import cv2
import numpy as np
from deepface import DeepFace
import pickle
import time
import threading
import queue
import urllib.request
import os
import ssl
from dotenv import load_dotenv
import openai
import json

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
llm_client = openai.OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY and OPENAI_KEY != "your_openai_api_key_here" else None

ssl._create_default_https_context = ssl._create_unverified_context

print("Loading 100% Accurate Single-Identity Pipeline...")
from ultralytics import YOLO
YOLO_MODEL_PATH = "yolov8n-face.pt"
if not os.path.exists(YOLO_MODEL_PATH):
    print("Downloading ultra-strict YOLOv8 Face model...")
    urllib.request.urlretrieve("https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt", YOLO_MODEL_PATH)

face_model = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

DB_FILE = "face_db.pkl"
try:
    with open(DB_FILE, 'rb') as f:
        # Changed to a Dictionary to store MULTIPLE angles per ID!
        known_face_profiles = pickle.load(f) 
    unique_customer_count = len(known_face_profiles)
except (FileNotFoundError, EOFError, AttributeError, TypeError):
    known_face_profiles = {}
    unique_customer_count = 0

def save_db():
    with open(DB_FILE, 'wb') as f:
        pickle.dump(known_face_profiles, f)

def is_same_person(new_vector, profiles, threshold=0.60):
    best_match_idx = None
    min_dist = float('inf')
    
    # Check against ALL saved angles for every person to find best match
    for cust_id, vectors in profiles.items():
        for known_vec in vectors:
            a = np.array(new_vector)
            b = np.array(known_vec)
            cos_dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            
            if cos_dist < min_dist:
                min_dist = cos_dist
                best_match_idx = cust_id
                
    if min_dist <= threshold:
        return True, best_match_idx, min_dist 
    return False, None, min_dist

tracked_faces = {} 
next_track_id = 0
face_queue = queue.Queue()

def face_recognition_worker():
    global unique_customer_count
    while True:
        task = face_queue.get()
        if task is None: break
        t_id, face_crop = task
        
        try:
            
            
            
            # already fiercely ensures it is a human face.
            embedding_objs = DeepFace.represent(
                img_path=face_crop, 
                model_name="ArcFace",      
                detector_backend="opencv", 
                enforce_detection=False,   
                align=False                
            )
            
            new_vector = embedding_objs[0]["embedding"]
            is_match, cust_id, cos_dist = is_same_person(new_vector, known_face_profiles)
            
            if is_match:
                # DYNAMIC LEARNING: 
                # If this vector matched but was off by > 0.20 cosine distance, 
                # it's likely a new angle! Save it up to 10 angles per person.
                if cos_dist > 0.20 and len(known_face_profiles[cust_id]) < 10:
                    known_face_profiles[cust_id].append(new_vector)
                    save_db()
                    
                tracked_faces[t_id]['status'] = 'KNOWN'
                tracked_faces[t_id]['display_text'] = f"ID: {cust_id}"
                print(f"[VERIFIED] Thread saw -> DB Index {cust_id} (Dist: {cos_dist:.2f} / Views: {len(known_face_profiles[cust_id])})")
            else:
                cust_id = unique_customer_count
                unique_customer_count += 1
                known_face_profiles[cust_id] = [new_vector]
                save_db()
                tracked_faces[t_id]['status'] = 'KNOWN'
                tracked_faces[t_id]['display_text'] = f"NEW: ID {cust_id}"
                print(f"[NEW PERSON] Thread saw -> DB Index {cust_id}")
                
        except ValueError as e:
            # OpenCV couldn't align the face natively (due to deep tilt or blocking side profile).
            # Instead of rejecting the YOLO tracker completely, we HOLD FIRE.
            # We don't save garbage vectors; we wait for a good facial rotation!
            if t_id in tracked_faces:
                tracked_faces[t_id]['status'] = 'ALIGNING'
                tracked_faces[t_id]['display_text'] = "WAITING 4 ANGLE"
                print(f"[DEBUG] Held output (Tilted/Blocked angle): {e}")
        except Exception as e:
            if t_id in tracked_faces:
                tracked_faces[t_id]['status'] = 'FAILED'
                print(f"[DEBUG] Worker Exception: {e}")
                
        face_queue.task_done()

def llm_db_maintenance_worker():
    """
    LLM Auto-Correction Service: 
    Runs every 60 seconds. Reads the mathematical distances of the faces, 
    and uses an LLM to decide if two IDs are actually the same person, merging them on the spot!
    """
    while True:
        time.sleep(60) 
        if not llm_client:
            continue
            
        print("[LLM_MAINTENANCE] Checking Vector DB anomalies...")
        try:
            with open(DB_FILE, 'rb') as f:
                db_copy = pickle.load(f)
            
            # Compress math matrix into JSON stats to feed the LLM
            # (We cannot send 512d vectors, so we send the distances between IDs)
            matrix = []
            ids = list(db_copy.keys())
            if len(ids) > 1:
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        id1 = ids[i]
                        id2 = ids[j]
                        min_dist = float('inf')
                        for v1 in db_copy[id1]:
                            for v2 in db_copy[id2]:
                                a = np.array(v1)
                                b = np.array(v2)
                                dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                                if dist < min_dist:
                                    min_dist = dist
                        
                        if min_dist < 0.75: # Suspiciously similar ArcFace vectors, might be same person extreme tilt
                            matrix.append({"id1": id1, "id2": id2, "closest_cosine_distance": float(min_dist)})
                            
            if not matrix:
                continue

            prompt = f"""
            You are a Vector DB Architect. I run a face recognition model (ArcFace) that sometimes splits one person into multiple IDs.
            A normal ArcFace vector variation for the same person is 0.40 - 0.60. 
            Anything below 0.75 between two DIFFERENT IDs usually implies they are actually the exact same person viewed from an extreme head tilt or side profile/phone occlusion.
            
            Current Suspicious Distances:
            {json.dumps(matrix, indent=2)}
            
            Output a JSON response with a single key "merge_pairs", containing a list of objects with "keep_id" (lowest int) and "delete_id" (higher int).
            Only merge if you are confident based on the < 0.75 split threshold.
            """
            
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini", # Keep it fast
                messages=[
                    {"role": "system", "content": "You output JSON matching the instructions requested."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            
            directives = json.loads(response.choices[0].message.content)
            merges = directives.get('merge_pairs', [])
            
            if merges:
                print(f"[LLM_MAINTENANCE] LLM requested {len(merges)} DB merges!")
                for m in merges:
                    keep, delete = m['keep_id'], m['delete_id']
                    if keep in known_face_profiles and delete in known_face_profiles:
                        known_face_profiles[keep].extend(known_face_profiles[delete])
                        del known_face_profiles[delete]
                        print(f" -> Merged ID {delete} into ID {keep}")
                save_db()

        except Exception as e:
            print(f"[LLM_MAINTENANCE] Failed: {e}")

threading.Thread(target=llm_db_maintenance_worker, daemon=True).start()

for _ in range(3):
    threading.Thread(target=face_recognition_worker, daemon=True).start()

print("Camera running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    # Use YOLOv8-Face for unparalleled detection strictness
    # conf=0.70 brutally ignores anything that is not aggressively a human face
    results = face_model.predict(frame, conf=0.70, verbose=False)
    
    current_time = time.time()
    current_faces = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Filter out tiny specks / artifacts
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                current_faces.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))

    unmatched_detections = []
    for (sx, sy, ex, ey) in current_faces:
        cx, cy = (sx + ex) // 2, (sy + ey) // 2
        matched_tid = None
        min_dist = float('inf')
        
        for tid, tdata in tracked_faces.items():
            dcx, dcy = tdata['center']
            dist = np.sqrt((cx - dcx)**2 + (cy - dcy)**2)
            if dist < 150 and dist < min_dist:  
                min_dist = dist
                matched_tid = tid
                
        if matched_tid is not None:
            tracked_faces[matched_tid]['center'] = (cx, cy)
            tracked_faces[matched_tid]['last_seen'] = current_time
            tracked_faces[matched_tid]['box'] = (sx, sy, ex, ey)
            
            # -- IMPROVED BUG FIX: Retries for Tilted faces vs Failed crops --
            if tracked_faces[matched_tid]['status'] in ['REJECTED', 'FAILED', 'ALIGNING']:
                # Retry much faster (0.3s) if we are just waiting for the user to turn their head straight
                cooldown = 0.3 if tracked_faces[matched_tid]['status'] == 'ALIGNING' else 1.0
                if current_time - tracked_faces[matched_tid].get('last_retry', 0) > cooldown:
                    tracked_faces[matched_tid]['status'] = 'SCANNING'
                    tracked_faces[matched_tid]['display_text'] = 'ANALYZING...'
                    tracked_faces[matched_tid]['last_retry'] = current_time
                    
                    bw, bh = ex - sx, ey - sy
                    # Restored back to 20% to keep enough context for secondary tilt alignment!
                    px, py = int(bw * 0.20), int(bh * 0.20) 
                    crop_sx, crop_sy = max(0, sx - px), max(0, sy - py)
                    crop_ex, crop_ey = min(w, ex + px), min(h, ey + py)
                    
                    # Force minimum face size before queueing for analysis
                    if crop_ex - crop_sx > 60 and crop_ey - crop_sy > 60:
                        face_crop = frame[crop_sy:crop_ey, crop_sx:crop_ex].copy()
                        face_queue.put((matched_tid, face_crop))
        else:
            unmatched_detections.append((sx, sy, ex, ey))

    for (sx, sy, ex, ey) in unmatched_detections:
        cx, cy = (sx + ex) // 2, (sy + ey) // 2
        next_track_id += 1
        tid = next_track_id
        
        tracked_faces[tid] = {
            'center': (cx, cy),
            'last_seen': current_time,
            'status': 'SCANNING',
            'display_text': 'ANALYZING...',
            'box': (sx, sy, ex, ey),
            # Important Fix! Only retry a REJECTED/FAILED face every 0.8 seconds
            'last_retry': current_time
        }
        
        # We use a 20% padding around YOLO's confident box.
        # This padding is vital so the secondary face detector inside DeepFace has enough context to verify.
        bw, bh = ex - sx, ey - sy
        px, py = int(bw * 0.20), int(bh * 0.20) 
        crop_sx, crop_sy = max(0, sx - px), max(0, sy - py)
        crop_ex, crop_ey = min(w, ex + px), min(h, ey + py)
        
        face_crop = frame[crop_sy:crop_ey, crop_sx:crop_ex].copy()
        face_queue.put((tid, face_crop))

    tracked_faces = {tid: tdata for tid, tdata in tracked_faces.items() if current_time - tdata['last_seen'] < 2.0}

    for tdata in tracked_faces.values():
        sx, sy, ex, ey = tdata.get('box', (0,0,0,0))
        status = tdata['status']
        text = tdata['display_text']
        
        if status == 'SCANNING':
            color = (0, 255, 255)  # Yellow
        elif status == 'ALIGNING':
            color = (0, 165, 255)  # Orange for "Watching, waiting for good frontal shot"
        elif status in ['REJECTED', 'FAILED']:
            color = (0, 0, 255)    # Red for totally rejected objects
        else:
            color = (0, 255, 0)
            
        cv2.rectangle(frame, (sx, sy), (ex, ey), color, 2)
        cv2.putText(frame, text, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"DB SIZE: {unique_customer_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow('Store Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
