import cv2
import numpy as np
from ultralytics import YOLO
import torch
import queue
import threading
import time
import pickle
import os
import ssl
from transformers import CLIPProcessor, CLIPModel

ssl._create_default_https_context = ssl._create_unverified_context

print("Loading CLIP-based Pipeline...")

# 1. Detection Models
body_model = YOLO("yolov8s.pt")
face_model = YOLO("yolov8n-face.pt")

# 2. CLIP Feature Extractor (Robust to lighting/pose)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Loading CLIP on {device}...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Threads & Tracking
detection_queue = queue.Queue(maxsize=3)
analysis_queue = queue.Queue()
result_coords = {}
active_tracks = {}
track_id_counter = 0

daily_identity_db = {}
next_customer_id = 0

# --- TIGHTENED THRESHOLDS ---
STRONG_MATCH_THRESHOLD = 0.94
MATCH_THRESHOLD = 0.91
BUFFER_THRESHOLD = 0.85

def tracking_worker():
    while True:
        task = detection_queue.get()
        if task is None: break
        tid, raw_frame = task
        
        body_results = body_model.predict(raw_frame, classes=[0], conf=0.50, imgsz=320, verbose=False)
        face_results = face_model.predict(raw_frame, conf=0.40, imgsz=320, verbose=False)
        
        faces = []
        for fr in face_results:
            for box in fr.boxes:
                fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                faces.append((fx1, fy1, fx2, fy2))
                
        bodies = []
        for r in body_results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                h, w = raw_frame.shape[:2]
                body_crop = raw_frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                
                if body_crop.shape[0] > 0 and body_crop.shape[1] > 0:
                    best_face = None
                    for f in faces:
                        fx1, fy1, fx2, fy2 = f
                        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                        if bx1 < fcx < bx2 and by1 < fcy < by1 + (by2 - by1) * 0.6:
                            best_face = f
                            break
                            
                    bodies.append(((bx1, by1, bx2, by2), best_face, body_crop.copy()))
                    if best_face is not None:
                        faces.remove(best_face)
        
        result_coords['bodies'] = bodies
        detection_queue.task_done()

threading.Thread(target=tracking_worker, daemon=True).start()

def extract_clip_features(body_crop, frame_to_draw_on=None, x_offset=0, y_offset=0):
    try:
        h, w = body_crop.shape[:2]
        top = int(h * 0.10)
        bottom = int(h * 0.75)
        left = int(w * 0.15)
        right = int(w * 0.85)
        if (bottom - top) < 30 or (right - left) < 30:
            return None
        torso_crop = body_crop[top:bottom, left:right]
        rgb_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2RGB)
        inputs = clip_processor(images=rgb_torso, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)

        if hasattr(features, "pooler_output"):
            features = features.pooler_output

        features = features.float()
        features = features / torch.norm(features, dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return None

def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

def determine_identity(body_crop, frame=None, x_offset=0, y_offset=0, tid=None):
    global next_customer_id, daily_identity_db, active_tracks
    
    # --- 1. TRACK-FIRST STICKINESS (SOFT LOCK) ---
    if tid is not None and tid in active_tracks:
        current_status = active_tracks[tid].get('status', '')
        current_id = active_tracks[tid].get('id', '...')
        lock_frames = active_tracks[tid].get('lock_frames', 0)
        
        # Keep lock alive for X frames, don't lock forever
        if current_status.startswith("Verified") and current_id != '...' and lock_frames > 0:
            active_tracks[tid]['lock_frames'] -= 1
            return current_id, f"Verified (Locked:{lock_frames})"
            
    # --- 2. EXTRACT EMBEDDING ---
    features = extract_clip_features(body_crop, frame, x_offset, y_offset)
    if features is None:
        return "Unknown", "NO FEATURES"
        
    # --- 3. TEMPORAL BUFFERING (Multi-frame decision) ---
    track_features = features
    if tid is not None and tid in active_tracks:
        if 'recent_embeddings' not in active_tracks[tid]:
            active_tracks[tid]['recent_embeddings'] = []
        if 'observing_frames' not in active_tracks[tid]:
            active_tracks[tid]['observing_frames'] = 0
            
        active_tracks[tid]['recent_embeddings'].append(features)
        active_tracks[tid]['observing_frames'] += 1
        
        # COLLECT FOR 5 FRAMES BEFORE AVERAGING
        if len(active_tracks[tid]['recent_embeddings']) < 5:
            return "...", "OBSERVING (Collecting)"
            
        if len(active_tracks[tid]['recent_embeddings']) > 5:
            active_tracks[tid]['recent_embeddings'].pop(0)
            
        avg_temporal = np.mean(active_tracks[tid]['recent_embeddings'], axis=0)
        track_features = avg_temporal / np.linalg.norm(avg_temporal)

    best_id = None
    best_sim = -1.0
    
    for cid, data in daily_identity_db.items():
        if len(data['features']) > 0:
            for known_vec in data['features']:
                sim = calculate_cosine_similarity(track_features, known_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_id = cid

    # --- 4. SPATIAL CONSISTENCY (Penalize if ID is clearly active elsewhere) ---
    if best_id is not None and best_sim > BUFFER_THRESHOLD:
        for active_tid, tdata in active_tracks.items():
            if active_tid != tid and tdata.get('id') == best_id:
                if tdata.get('frames_since_analysis', 0) < 30: 
                    # Someone else on screen already vigorously claims this ID!
                    print(f"[PENALTY] ID {best_id} matches {best_sim:.3f} but is ACTIVE elsewhere. Penalizing -0.15.")
                    best_sim -= 0.15 
                    break

    if best_id is not None:
        # STRONG MATCH PRIORITY
        if best_sim > STRONG_MATCH_THRESHOLD:
            print(f"[MATCH] Confirmed ID {best_id} with STRONG similarity {best_sim:.3f}")
            if len(daily_identity_db[best_id]['features']) < 20: 
                daily_identity_db[best_id]['features'].append(track_features)
            with open('clothes_db.pkl', 'wb') as f:
                pickle.dump(daily_identity_db, f)
            if tid is not None and tid in active_tracks:
                active_tracks[tid]['lock_frames'] = 15 # Lock heavily!
            return best_id, "Verified"
            
        # STANDARD MATCH
        elif best_sim > MATCH_THRESHOLD:
            print(f"[MATCH] Confirmed ID {best_id} with similarity {best_sim:.3f}")
            if len(daily_identity_db[best_id]['features']) < 20: 
                daily_identity_db[best_id]['features'].append(track_features)
            with open('clothes_db.pkl', 'wb') as f:
                pickle.dump(daily_identity_db, f)
            if tid is not None and tid in active_tracks:
                active_tracks[tid]['lock_frames'] = 10 # Lock moderately!
            return best_id, "Verified"
            
        # BUFFER WAIT
        elif best_sim > BUFFER_THRESHOLD:
            print(f"[WAIT] Buffer zone for best candidate ID {best_id} (sim {best_sim:.3f})")
            return "...", "OBSERVING"

    # --- 5. NEW ID CREATION DELAY ---
    # Do not spawn a new ID unless we have helplessly observed for 15 full frames
    if tid is not None and active_tracks[tid].get('observing_frames', 0) < 15:
        return "...", f"WAITING ({active_tracks[tid].get('observing_frames')}/15)"

    # SPAWN NEW ID (only after failing buffer and observing sufficiently)
    current_id = next_customer_id
    daily_identity_db[current_id] = {'features': [track_features]}
    next_customer_id += 1
    
    print(f"[NEW ID] Created new ID {current_id} (best sim was {best_sim:.3f})")
    with open('clothes_db.pkl', 'wb') as f:
        pickle.dump(daily_identity_db, f)
        
    return current_id, "NEW OUTFIT"

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0: return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(area1 + area2 - intersection)

def background_analysis_worker():
    while True:
        task = analysis_queue.get()
        if task is None: break
        tid, body_crop, frame, bx1, by1 = task
        
        cid, method = determine_identity(body_crop, frame, bx1, by1, tid)
        
        if tid in active_tracks:
            active_tracks[tid]['id'] = cid
            active_tracks[tid]['status'] = method
            
        analysis_queue.task_done()

threading.Thread(target=background_analysis_worker, daemon=True).start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting Camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    if detection_queue.empty():
        detection_queue.put((0, frame.copy()))
        
    detected_bodies = result_coords.get('bodies', [])
    
    for (body_box, face_box, body_crop) in detected_bodies:
        bx1, by1, bx2, by2 = body_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        if face_box is not None:
            gx1, gy1, gx2, gy2 = face_box
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        matched_tid = None
        best_iou = 0.0
        min_dist = float('inf')
        
        for tid, tdata in active_tracks.items():
            if 'box' in tdata:
                iou = calculate_iou((bx1, by1, bx2, by2), tdata['box'])
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    matched_tid = tid

        if matched_tid is None:
            for tid, tdata in active_tracks.items():
                last_cx, last_cy = tdata['center']
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    matched_tid = tid
                
        if matched_tid is None:
            track_id_counter += 1
            matched_tid = track_id_counter
            active_tracks[matched_tid] = {
                'center': (cx, cy), 
                'box': (bx1, by1, bx2, by2),
                'id': '...', 
                'status': 'ANALYZING', 
                'last_seen': time.time(),
                'frames_since_analysis': 0,
                'lock_frames': 0,
                'observing_frames': 0
            }
            analysis_queue.put((matched_tid, body_crop, frame.copy(), bx1, by1))
        else:
            active_tracks[matched_tid]['center'] = (cx, cy)
            active_tracks[matched_tid]['box'] = (bx1, by1, bx2, by2)
            active_tracks[matched_tid]['last_seen'] = time.time()
            active_tracks[matched_tid]['frames_since_analysis'] += 1
            
            # ALWAYS allow re-analyzing based on time to feed the Soft Lock and observing loops
            if active_tracks[matched_tid]['frames_since_analysis'] > 10:
                active_tracks[matched_tid]['frames_since_analysis'] = 0
                analysis_queue.put((matched_tid, body_crop, frame.copy(), bx1, by1))
            
        display_id = active_tracks[matched_tid].get('id', '?')
        status = active_tracks[matched_tid].get('status', 'WAIT')
        cv2.putText(frame, f"ID {display_id}: {status}", (bx1, by2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    curr_time = time.time()
    active_tracks = {k: v for k, v in active_tracks.items() if curr_time - v['last_seen'] < 2.0}

    cv2.putText(frame, f"TODAYS CUSTOMERS: {next_customer_id}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Multi-Modal Tri-Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
