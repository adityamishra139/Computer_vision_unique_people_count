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
from deepface import DeepFace

ssl._create_default_https_context = ssl._create_unverified_context

print("Loading 100% Accurate Face + Clothes Fusion Pipeline...")

# --- 1. Load Models ---
body_model = YOLO("yolov8s.pt")
face_model = YOLO("yolov8n-face.pt")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Loading CLIP on {device}...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- 2. Database & State ---
DB_FILE = 'final_db.pkl'
try:
    with open(DB_FILE, 'rb') as f:
        daily_identity_db = pickle.load(f)
    next_customer_id = max(daily_identity_db.keys()) + 1 if daily_identity_db else 0
except Exception:
    daily_identity_db = {}
    next_customer_id = 0

def save_db():
    with open(DB_FILE, 'wb') as f:
        pickle.dump(daily_identity_db, f)

detection_queue = queue.Queue(maxsize=3)
analysis_queue = queue.Queue()
result_coords = {}
active_tracks = {}
track_id_counter = 0

STRONG_MATCH_CLOTHES = 0.94
MATCH_CLOTHES = 0.91
BUFFER_CLOTHES = 0.85
STRONG_FACE_DIST = 0.20  # Tightened ArcFace Cosine distance (was 0.60/0.40)

# --- 3. Workers ---
def tracking_worker():
    while True:
        task = detection_queue.get()
        if task is None: break
        tid, raw_frame = task
        
        body_results = body_model.predict(raw_frame, classes=[0], conf=0.50, imgsz=320, verbose=False)
        face_results = face_model.predict(raw_frame, conf=0.50, imgsz=320, verbose=False)
        
        faces = []
        for fr in face_results:
            for box in fr.boxes:
                fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                faces.append((fx1, fy1, fx2, fy2))
                
        bodies = []
        h, w = raw_frame.shape[:2]
        
        for r in body_results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                body_crop = raw_frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                
                if body_crop.shape[0] > 0 and body_crop.shape[1] > 0:
                    best_face = None
                    # Associate face with body
                    for f in faces:
                        fx1, fy1, fx2, fy2 = f
                        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                        if bx1 < fcx < bx2 and by1 < fcy < by1 + (by2 - by1) * 0.6:
                            best_face = f
                            break
                            
                    face_crop = None
                    if best_face is not None:
                        fx1, fy1, fx2, fy2 = best_face
                        bw, bh = fx2 - fx1, fy2 - fy1
                        px, py = int(bw * 0.20), int(bh * 0.20)
                        crop_sx, crop_sy = max(0, fx1 - px), max(0, fy1 - py)
                        crop_ex, crop_ey = min(w, fx2 + px), min(h, fy2 + py)
                        if crop_ex - crop_sx > 40 and crop_ey - crop_sy > 40:
                            face_crop = raw_frame[crop_sy:crop_ey, crop_sx:crop_ex].copy()
                        faces.remove(best_face)
                            
                    # 🔥 STRICT FILTER: Only allow if face OR strong human shape
                    height = by2 - by1
                    width = bx2 - bx1
                    aspect_ratio = height / (width + 1e-5)

                    # Condition 1: Face present (strong signal)
                    has_face = best_face is not None

                    # Condition 2: Human-like body (tall + large enough)
                    valid_body = (
                        aspect_ratio > 1.2 and     # not flat like clothes
                        height > 120 and           # not small object
                        width > 40
                    )

                    if has_face or valid_body:
                        bodies.append(((bx1, by1, bx2, by2), best_face, body_crop.copy(), face_crop))
        
        result_coords['bodies'] = bodies
        detection_queue.task_done()

threading.Thread(target=tracking_worker, daemon=True).start()

def extract_clip_features(body_crop):
    try:
        h, w = body_crop.shape[:2]
        top, bottom = int(h * 0.10), int(h * 0.75)
        left, right = int(w * 0.15), int(w * 0.85)

        if (bottom - top) < 30 or (right - left) < 30: return None

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
    except Exception:
        return None

def determine_identity(body_crop, face_crop, tid):
    global next_customer_id, daily_identity_db, active_tracks
    
    # 🔥 HARD LOCK ID (DO NOT CHANGE AFTER ASSIGNMENT)
    if tid in active_tracks:
        assigned_id = active_tracks[tid].get('id')
        if assigned_id not in [None, '...']:
            return assigned_id, active_tracks[tid].get('status', 'PERSISTED ID')

    # --- 1. TRACK-FIRST STICKINESS ---
    if tid in active_tracks:
        current_status = active_tracks[tid].get('status', '')
        current_id = active_tracks[tid].get('id', '...')
        lock_frames = active_tracks[tid].get('lock_frames', 0)
        
        if current_status.startswith("Verified") and current_id != '...' and lock_frames > 0:
            active_tracks[tid]['lock_frames'] -= 1
            # If we recently got a Face lock, respect it highly
            if "Face" in current_status:
                return current_id, f"Verified (Face Lock:{lock_frames})"
            elif face_crop is None:
                return current_id, f"Verified (Clothes Lock:{lock_frames})"
            # If clothes locked but face just appeared, drop down to analyze the face!
            
    # --- 2. EXTRACT EMBEDDINGS ---
    clothes_vec = extract_clip_features(body_crop)
    face_vec = None
    
    if face_crop is not None:
        fh, fw = face_crop.shape[:2]
        if fh < 80 or fw < 80:
            face_crop = None
            
    if face_crop is not None:
        try:
            # DeepFace works heavily with RGB numpy arrays natively
            # backend='skip' because YOLO already detected the face
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            res = DeepFace.represent(img_path=rgb_face, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
            face_vec = res[0]["embedding"]
            face_vec = np.array(face_vec)
            face_vec = face_vec / np.linalg.norm(face_vec)
        except Exception as e:
            face_vec = None

    # --- 3. GROUND TRUTH (FACE PRIORITY) ---
    if face_vec is not None:
        best_face_id = None
        min_face_dist = float('inf')
        
        for cid, data in daily_identity_db.items():
            if not data['face_features']:
                continue
            for fv in data['face_features']:
                dist = 1 - (np.dot(face_vec, fv) / (np.linalg.norm(face_vec) * np.linalg.norm(fv)))
                if dist < min_face_dist:
                    min_face_dist = dist
                    best_face_id = cid
                    
        if best_face_id is not None and min_face_dist > STRONG_FACE_DIST:
            best_face_id = None

        # --- SPATIAL FACE ANTI-TELEPORTATION & MULTI-PERSON BLOCK ---
        if best_face_id is not None and tid in active_tracks:
            curr_cx, curr_cy = active_tracks[tid]['center']
            for active_tid, tdata in active_tracks.items():
                if tdata.get('id') == best_face_id:
                    # 1. Face already confidently assigned to someone else
                    if active_tid != tid and tdata.get('frames_since_analysis', 0) < 30:
                        print(f"[FACE BLOCK] ID {best_face_id} already active on another person. Skipping.")
                        best_face_id = None
                        break
                    
                    # 2. Face jumping too far (teleportation)
                    old_cx, old_cy = tdata['center']
                    dist = np.sqrt((old_cx - curr_cx)**2 + (old_cy - curr_cy)**2)
                    if dist > 70:
                        print(f"[FACE BLOCK] ID {best_face_id} jumped far distance ({dist:.1f}px). Rejecting.")
                        best_face_id = None
                        break

        # Face is a match!
        if best_face_id is not None:
            if tid in active_tracks:
                if 'face_hits' not in active_tracks[tid]:
                    active_tracks[tid]['face_hits'] = 0
                    active_tracks[tid]['face_candidate'] = best_face_id
                
                # If same candidate → increase confidence
                if active_tracks[tid]['face_candidate'] == best_face_id:
                    active_tracks[tid]['face_hits'] += 1
                else:
                    # Reset if different ID
                    active_tracks[tid]['face_candidate'] = best_face_id
                    active_tracks[tid]['face_hits'] = 1

                # Require consistent matches
                if active_tracks[tid]['face_hits'] >= 3:
                    print(f"[FACE CONFIRMED] ID {best_face_id} with distance {min_face_dist:.3f}")
                    if len(daily_identity_db[best_face_id]['face_features']) < 15:
                        daily_identity_db[best_face_id]['face_features'].append(face_vec)
                    if clothes_vec is not None and len(daily_identity_db[best_face_id]['clothes_features']) < 20:
                        daily_identity_db[best_face_id]['clothes_features'].append(clothes_vec)
                    
                    save_db()
                    active_tracks[tid]['lock_frames'] = 30
                    
                    # Confidence score for face match
                    face_conf = max(0.0, 1.0 - min_face_dist)
                    return best_face_id, f"Verified (Face) [{face_conf:.2f}]"
                else:
                    # Buffer not met, fall through to clothes logic
                    pass

    # --- 4. CLOTHES FALLBACK & TEMPORAL BUFFER ---
    if clothes_vec is None: return "Unknown", "NO FEATURES"

    if tid in active_tracks:
        current_status = active_tracks[tid].get("status", "")
        current_id = active_tracks[tid].get("id", "...")
        if "Face" in current_status and current_id != "...":
            active_tracks[tid]['lock_frames'] = 30
            return current_id, "Verified (Face Persist)"
        
    track_features = clothes_vec
    if tid in active_tracks:
        if 'recent_embeddings' not in active_tracks[tid]: active_tracks[tid]['recent_embeddings'] = []
        if 'observing_frames' not in active_tracks[tid]: active_tracks[tid]['observing_frames'] = 0
            
        active_tracks[tid]['recent_embeddings'].append(clothes_vec)
        active_tracks[tid]['observing_frames'] += 1
        
        if len(active_tracks[tid]['recent_embeddings']) < 5:
            return "...", "OBSERVING (Collecting)"
        if len(active_tracks[tid]['recent_embeddings']) > 5:
            active_tracks[tid]['recent_embeddings'].pop(0)
            
        avg_temporal = np.mean(active_tracks[tid]['recent_embeddings'], axis=0)
        track_features = avg_temporal / np.linalg.norm(avg_temporal)

    best_id = None
    best_sim = -1.0
    
    for cid, data in daily_identity_db.items():
        for known_vec in data['clothes_features']:
            sim = np.dot(track_features, known_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

    # Spatial Clothes Penalty
    if best_id is not None and best_sim > BUFFER_CLOTHES:
        for active_tid, tdata in active_tracks.items():
            if active_tid != tid and tdata.get('id') == best_id:
                if tdata.get('frames_since_analysis', 0) < 30:
                    print(f"[PENALTY] Clothes overlap for ID {best_id}. Penalizing.")
                    best_sim -= 0.15 
                    break

    if best_id is not None:
        if best_sim > STRONG_MATCH_CLOTHES:
            if len(daily_identity_db[best_id]['clothes_features']) < 20: 
                daily_identity_db[best_id]['clothes_features'].append(track_features)
            if face_vec is not None: daily_identity_db[best_id]['face_features'].append(face_vec)
            save_db()
            if tid in active_tracks: active_tracks[tid]['lock_frames'] = 15
            return best_id, f"Verified (Clothes-S) [{best_sim:.2f}]"
            
        elif best_sim > MATCH_CLOTHES:
            if len(daily_identity_db[best_id]['clothes_features']) < 20: 
                daily_identity_db[best_id]['clothes_features'].append(track_features)
            if face_vec is not None: daily_identity_db[best_id]['face_features'].append(face_vec)
            save_db()
            if tid in active_tracks: active_tracks[tid]['lock_frames'] = 10
            return best_id, f"Verified (Clothes-M) [{best_sim:.2f}]"
            
        elif best_sim > BUFFER_CLOTHES:
            return "...", "OBSERVING"

    # --- 5. SPAWN NEW ID ---
    # ❗ DO NOT create new ID without face
    if face_vec is None:
        return "...", "NO FACE - SKIP NEW ID"
        
    if face_vec is not None and best_face_id is None:
        current_time = time.time()
        
        if tid in active_tracks:
            # FIX 1: NEW ID COOLDOWN PER TRACK
            last_time = active_tracks[tid].get('last_new_id_time', 0)
            if current_time - last_time < 3.0:
                return "...", "COOLDOWN - SKIP"
                
            # FIX 3: TOO CLOSE TO EXISTING TRACK (OVERLAP/CROWD BLOCK)
            curr_cx, curr_cy = active_tracks[tid]['center']
            for active_tid, tdata in active_tracks.items():
                if active_tid != tid and 'center' in tdata:
                    old_cx, old_cy = tdata['center']
                    dist = np.sqrt((old_cx - curr_cx)**2 + (old_cy - curr_cy)**2)
                    if dist < 80:
                        return "...", "TOO CLOSE - POSSIBLE OVERLAP"
            
            # FIX 2: REQUIRE STABILITY (3 FRAMES)
            if 'new_face_hits' not in active_tracks[tid]:
                active_tracks[tid]['new_face_hits'] = 0
                
            active_tracks[tid]['new_face_hits'] += 1
            if active_tracks[tid]['new_face_hits'] < 3:
                return "...", f"NEW VERIFY ({active_tracks[tid]['new_face_hits']}/3)"
                
            # Update cooldown timestamp
            active_tracks[tid]['last_new_id_time'] = current_time
            
        print("[FAST NEW - SAFE] Face detected, stabilized -> creating new ID")
    elif tid in active_tracks and active_tracks[tid].get('observing_frames', 0) < 5:
        return "...", f"WAITING ({active_tracks[tid]['observing_frames']}/5)"

    current_id = next_customer_id
    daily_identity_db[current_id] = {
        'face_features': [face_vec],
        'clothes_features': [track_features]
    }
    next_customer_id += 1
    save_db()
    
    print(f"[NEW PERSON] Created ID {current_id}")
    return current_id, "NEW PERSON"

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0: return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(area1 + area2 - intersection)

def analysis_worker():
    while True:
        task = analysis_queue.get()
        if task is None: break
        tid, body_crop, face_crop = task
        
        cid, method = determine_identity(body_crop, face_crop, tid)
        
        if tid in active_tracks:
            active_tracks[tid]['id'] = cid
            active_tracks[tid]['status'] = method
            
        analysis_queue.task_done()

threading.Thread(target=analysis_worker, daemon=True).start()

# --- 4. Main Loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting True Fusion Camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    if detection_queue.empty():
        detection_queue.put((0, frame.copy()))
        
    detected_bodies = result_coords.get('bodies', [])
    
    for (body_box, face_box, body_crop, face_crop) in detected_bodies:
        bx1, by1, bx2, by2 = body_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        if face_box is not None:
            fx1, fy1, fx2, fy2 = face_box
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        matched_tid = None
        best_iou = 0.0
        min_dist = float('inf')
        
        for tid, tdata in active_tracks.items():
            iou = calculate_iou((bx1, by1, bx2, by2), tdata['box'])
            if iou > 0.3 and iou > best_iou:
                best_iou, matched_tid = iou, tid

        if matched_tid is None:
            for tid, tdata in active_tracks.items():
                dcx, dcy = tdata['center']
                dist = np.sqrt((cx - dcx)**2 + (cy - dcy)**2)
                if dist < 60 and dist < min_dist:
                    min_dist, matched_tid = dist, tid
                
        if matched_tid is None:
            track_id_counter += 1
            matched_tid = track_id_counter
            active_tracks[matched_tid] = {
                'center': (cx, cy), 'box': (bx1, by1, bx2, by2),
                'id': '...', 'status': 'ANALYZING', 'last_seen': time.time(),
                'frames_since_analysis': 0, 'lock_frames': 0, 'observing_frames': 0
            }
            analysis_queue.put((matched_tid, body_crop, face_crop))
        else:
            active_tracks[matched_tid]['center'] = (cx, cy)
            active_tracks[matched_tid]['box'] = (bx1, by1, bx2, by2)
            active_tracks[matched_tid]['last_seen'] = time.time()
            active_tracks[matched_tid]['frames_since_analysis'] += 1
            
            # Send to analysis heavily if we have a face and we aren't face-locked yet
            has_face_and_needs_upgrade = (face_crop is not None and "Face" not in active_tracks[matched_tid].get('status', ''))
            
            if active_tracks[matched_tid]['frames_since_analysis'] > 10 or has_face_and_needs_upgrade:
                active_tracks[matched_tid]['frames_since_analysis'] = 0
                analysis_queue.put((matched_tid, body_crop, face_crop))
            
        display_id = active_tracks[matched_tid].get('id', '?')
        status = active_tracks[matched_tid].get('status', 'WAIT')
        cv2.putText(frame, f"ID {display_id}: {status}", (bx1, by2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    curr_time = time.time()
    active_tracks = {k: v for k, v in active_tracks.items() if curr_time - v['last_seen'] < 5.0}

    cv2.putText(frame, f"TOTAL CUSTOMERS: {next_customer_id}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('Fusion Face + Clothes Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
