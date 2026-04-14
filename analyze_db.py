import pickle
import numpy as np

DB_FILE = "face_db.pkl"
try:
    with open(DB_FILE, 'rb') as f:
        db = pickle.load(f)
        
    print("==================================================")
    print(f"📊 VECTOR DATABASE ANALYSIS")
    print("==================================================")
    print(f"Total Unique IDs registered: {len(db)}\n")
    
    print("--- INTERNAL DISTANCES (Same Person Angles) ---")
    print("Ideal: < 0.20 (Very similar), Acceptable: < 0.40")
    for cust_id, vectors in db.items():
        print(f"\n👤 ID {cust_id} has {len(vectors)} saved visual profile(s).")
        if len(vectors) > 1:
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    a = np.array(vectors[i])
                    b = np.array(vectors[j])
                    cos_dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    
                    warning = "⚠️ DANGER (High Variance)" if cos_dist >= 0.35 else "✅ Safe"
                    print(f"  Angle {i} vs Angle {j} -> Cosine Distance: {cos_dist:.4f} {warning}")
                    
    print("\n--- CROSS-ID DISTANCES (Different People) ---")
    print("Ideal: > 0.60 (Clearly different people)")
    ids = list(db.keys())
    if len(ids) > 1:
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                min_dist = float('inf')
                for v1 in db[id1]:
                    for v2 in db[id2]:
                         a = np.array(v1)
                         b = np.array(v2)
                         dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                         if dist < min_dist:
                             min_dist = dist
                             
                warning = "🚨 COLLISION RISK (Too Similar)" if min_dist <= 0.45 else "✅ Safe"
                print(f"  Closest gap between ID {id1} and ID {id2}: {min_dist:.4f} {warning}")

except FileNotFoundError:
    print("No database found. Run the camera script first.")
except Exception as e:
    print("Error analyzing db:", e)
