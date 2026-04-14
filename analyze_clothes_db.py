import pickle
import numpy as np
import cv2
import requests

DB_FILE = "clothes_db.pkl"
LLM_API_URL = "http://localhost:11434/api/generate"

# Required for ORB Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def calculate_multi_similarity(vec1_tuple, vec2_tuple):
    osnet1, hist1, desc1 = vec1_tuple
    osnet2, hist2, desc2 = vec2_tuple
    
    # 1. OSNet Distance (Cosine Distance)
    osnet_dist = 1 - (np.dot(osnet1, osnet2) / (np.linalg.norm(osnet1) * np.linalg.norm(osnet2)))
    
    # 2. Color Distance (Bhattacharyya)
    color_dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    # 3. Nano-Dots Matching
    orb_matches = 0
    if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
        try:
            matches = bf.match(desc1, desc2)
            good_matches = [m for m in matches if m.distance < 50]
            orb_matches = len(good_matches)
        except Exception:
            pass
            
    return osnet_dist, color_dist, orb_matches

def ask_llm_for_tuning_advice(log_summary):
    prompt = f"""
    You are a Computer Vision Engineer debugging a Multi-Modal Clothes Re-Identification pipeline.
    The pipeline combines OSNet Structural Distance (0=identical), HSV Bhattacharyya Color Distance (0=identical), and ORB Nano-Dots matching (higher=better).
    
    Here is a statistical summary of the actual saved embeddings in the database across different IDs:
    
    {log_summary}
    
    Please analyze these statistics. 
    1. Identify any "collisions" where different individuals (cross-ID) have extremely close distances.
    2. Identify any "fragmentation" where the same individual (internal variance) has very high distances.
    3. Suggest EXACT strict thresholds for OSNet, Color, and ORB dots to prevent these errors going forward.

    Keep your feedback technical, actionable, and focus entirely on tuning the mathematical thresholds.
    """
    try:
        print("\n[LLM DIAGNOSTIC] Sending data to LLM for threshold tuning advice...")
        res = requests.post(LLM_API_URL, json={
            "model": "llama3", 
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        if res.status_code == 200:
            print("\n----- ✨ LLM ANALYSIS ✨ -----")
            print(res.json().get('response', '').strip())
            print("------------------------------\n")
        else:
            print(f"LLM Error: {res.status_code}")
    except Exception as e:
        print("LLM Offline or unavailable for diagnostic:", e)

try:
    with open(DB_FILE, 'rb') as f:
        db = pickle.load(f)
        
    print("==================================================")
    print(f"👕 CLOTHES VECTOR DATABASE ANALYSIS")
    print("==================================================")
    print(f"Total Unique Clothes IDs registered: {len(db)}\n")
    
    log_summary = ""
    
    print("--- INTERNAL DISTANCES (Same Person, Different Frames) ---")
    log_summary += "INTERNAL VARIANCES (Same ID):\n"
    for cust_id, data in db.items():
        vectors = data['features']
        print(f"\n👤 ID {cust_id} has {len(vectors)} saved outfit visual profiles.")
        if len(vectors) > 1:
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    o_dist, c_dist, matches = calculate_multi_similarity(vectors[i], vectors[j])
                    
                    # Ideal: o_dist < 0.45, c_dist < 0.30
                    warning = "⚠️ DANGER (High Variance)" if (o_dist >= 0.50 or c_dist >= 0.40) else "✅ Safe"
                    stat_line = f"  ID {cust_id}: Angle {i} vs {j} -> OSNet: {o_dist:.3f} | Color: {c_dist:.3f} | Dots: {matches} {warning}"
                    print(stat_line)
                    log_summary += stat_line + "\n"
                    
    print("\n--- CROSS-ID DISTANCES (Different People) ---")
    log_summary += "\nCROSS-ID VARIANCES (Different IDs):\n"
    ids = list(db.keys())
    if len(ids) > 1:
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                
                # Find closest gap between two distinct IDs to find "collision risks"
                best_o, best_c, best_m = 1.0, 1.0, 0
                for v1 in db[id1]['features']:
                    for v2 in db[id2]['features']:
                        o_dist, c_dist, matches = calculate_multi_similarity(v1, v2)
                        
                        # We track the "most similar" instance between two distinct IDs
                        if (o_dist + c_dist) < (best_o + best_c):
                            best_o, best_c, best_m = o_dist, c_dist, matches
                             
                warning = "🚨 COLLISION RISK (Too Similar)" if (best_o <= 0.40 and best_c <= 0.30) else "✅ Safe"
                stat_line = f"  Closest gap ID {id1} vs ID {id2}: OSNet: {best_o:.3f} | Color: {best_c:.3f} | Dots: {best_m} {warning}"
                print(stat_line)
                log_summary += stat_line + "\n"

    # Kick off the LLM Review
    ask_llm_for_tuning_advice(log_summary)

except FileNotFoundError:
    print("No database found. Run 'clothes_analyzer.py' with multiple people first to generate 'clothes_db.pkl'.")
except Exception as e:
    print("Error analyzing db:", e)