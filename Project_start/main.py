import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

print("Loading SOTA Tracking Model...")
model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(0)

unique_customer_count = 0

previous_x_positions = {}
id_lifespan = {} 
needs_face_scan = set()           
successfully_scanned_ids = set()  

known_face_vectors = []

# --- UPGRADED STRICT MATHEMATICS ---
# Using Cosine Similarity. For ArcFace, a match usually sits above 0.70.
# We set it to 0.68. If it's less than this, they are definitively a NEW person.
def is_same_person(new_vector, known_vectors, threshold=0.68):
    for idx, known_vec in enumerate(known_vectors):
        dot_product = np.dot(new_vector, known_vec)
        norm_a = np.linalg.norm(new_vector)
        norm_b = np.linalg.norm(known_vec)
        similarity = dot_product / (norm_a * norm_b)
        
        if similarity > threshold:
            return True, idx 
    return False, None

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width, _ = frame.shape
    tripwire_x = int(width / 2)
    scan_zone_end = tripwire_x + 300 

    cv2.line(frame, (tripwire_x, 0), (tripwire_x, height), (255, 0, 0), 2)
    cv2.line(frame, (scan_zone_end, 0), (scan_zone_end, height), (0, 100, 255), 1)
    cv2.putText(frame, "ENTRANCE", (tripwire_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "SCAN ZONE", (tripwire_x + 10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    # YOLO Tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.50, stream=True, device='mps')

    for r in results:
        boxes = r.boxes 
        for box in boxes:
            if box.id is not None:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                id_lifespan[track_id] = id_lifespan.get(track_id, 0) + 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if id_lifespan[track_id] > 5: # Reduced to 5 so it catches people faster in a crowd
                    if track_id not in previous_x_positions:
                        previous_x_positions[track_id] = center_x
                    else:
                        prev_x = previous_x_positions[track_id]
                        
                        if prev_x < tripwire_x and center_x >= tripwire_x:
                            needs_face_scan.add(track_id)

                        # --- MULTI-PERSON SCANNING LOGIC ---
                        if track_id in needs_face_scan and track_id not in successfully_scanned_ids:
                            
                            if center_x < scan_zone_end:
                                cv2.putText(frame, "SCANNING...", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                
                                # Add padding to the crop so RetinaFace can see the whole head easily
                                pad = 20
                                crop_y1, crop_y2 = max(0, y1 - pad), min(height, y2 + pad)
                                crop_x1, crop_x2 = max(0, x1 - pad), min(width, x2 + pad)
                                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                
                                try:
                                    # UPGRADE 1: model_name="ArcFace" (Utmost Accuracy for Identity)
                                    # UPGRADE 2: detector_backend="retinaface" (Utmost Accuracy for Crowd/Angle Face Detection)
                                    embedding_objs = DeepFace.represent(
                                        img_path=person_crop, 
                                        model_name="ArcFace", 
                                        detector_backend="retinaface", 
                                        enforce_detection=True
                                    )
                                    
                                    new_vector = embedding_objs[0]["embedding"]
                                    
                                    is_match, cust_id = is_same_person(new_vector, known_face_vectors)
                                    
                                    if is_match:
                                        print(f"ID {track_id} is Returning Customer (Matched with DB Index: {cust_id})")
                                    else:
                                        unique_customer_count += 1
                                        known_face_vectors.append(new_vector)
                                        print(f"✅ NEW UNIQUE CUSTOMER! (ID {track_id}) Total Unique: {unique_customer_count}")
                                    
                                    successfully_scanned_ids.add(track_id)
                                    
                                except Exception as e:
                                    # RetinaFace failed to find a face (looking away, blocked). Try again next frame!
                                    pass
                            
                            else:
                                successfully_scanned_ids.add(track_id)
                                print(f"❌ ID {track_id} left scan zone. Face missed.")

                        previous_x_positions[track_id] = center_x

                if track_id in successfully_scanned_ids:
                    cv2.putText(frame, "VERIFIED", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"UNIQUE CUSTOMERS: {unique_customer_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imshow('Store Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()