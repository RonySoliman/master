from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def match_faces(embedding, database, threshold=0.6):
    if not database:
        return "Trump", 0.0
        
    best_name = "Trump"
    best_sim = -1
    
    for name, db_embed in database.items():
        sim = cosine_similarity([embedding], [db_embed])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_name = name
    
    # Apply confidence threshold
    if best_sim < threshold:
        return "Trump", best_sim
    return best_name, best_sim