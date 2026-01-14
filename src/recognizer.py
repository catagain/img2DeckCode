import cv2
import numpy as np
import json
import imagehash
from PIL import Image
from pathlib import Path
from datetime import datetime

class CardRecognizer:
    # Class Constant: Define the resampling method for image scaling
    try:
        RESAMPLE_METHOD = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE_METHOD = Image.LANCZOS

    def __init__(self, confidence_threshold=90):
        self.threshold = confidence_threshold
        # Pre-load databases into memory for faster response in web environments
        self.id_db, self.hash_db = self._load_databases()
        
        # Optimized artwork cropping ratios
        self.art_ratios = {
            'ay1': 0.22, 'ay2': 0.69,
            'ax1': 0.18, 'ax2': 0.86
        }

    def _load_databases(self):
        """Loads JSON databases from the project's data folder."""
        script_dir = Path(__file__).parent.absolute()
        data_folder = script_dir.parent / 'data'
        try:
            db_path = data_folder / 'id_to_card_data.json'
            hash_path = data_folder / 'card_hashes.json'
            if not db_path.exists() or not hash_path.exists():
                print(f"[!] Warning: Database files not found in {data_folder}")
                return {}, {}
            with open(db_path, 'r', encoding='utf-8') as f:
                id_db = json.load(f)
            with open(hash_path, 'r', encoding='utf-8') as f:
                hash_db = json.load(f)
            return id_db, hash_db
        except Exception as e:
            print(f"[!] Database Initialization Error: {e}")
            return {}, {}

    def process_image(self, image_data, user_id="default", save_debug=True):
        """
        Main entry point: Processes the image and decides whether to output debug resources.
        image_data: File path (str/Path) or OpenCV image array (ndarray).
        save_debug: If True, saves crops and logs to output/{user_id}/{timestamp}/
        """
        if isinstance(image_data, (str, Path)):
            image = cv2.imread(str(image_data))
            img_filename = Path(image_data).stem
        else:
            image = image_data
            img_filename = "upload"

        if image is None: 
            return {"error": "Invalid image input"}

        # Prepare Debug Directory (Format: output/{user_id}/{img_name}_{timestamp}/)
        debug_path = None
        if save_debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_root = Path(__file__).parent.parent.absolute()
            debug_path = project_root / 'output' / user_id / f"{img_filename}_{timestamp}"
            debug_path.mkdir(parents=True, exist_ok=True)

        # Execute Detection and Matching
        results, debug_canvas = self._detect_and_match(image, debug_path)
        
        # Format Final Output
        final_output = self._format_output(results, user_id)

        # Save summary and grid image if debug mode is on
        if save_debug and debug_path:
            cv2.imwrite(str(debug_path / "_full_grid.jpg"), debug_canvas)
            with open(debug_path / "result.json", "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=4, ensure_ascii=False)
            print(f"[*] Debug resources collected at: {debug_path}")

        return final_output

    def _detect_and_match(self, image, debug_path=None):
        img_h, img_w = image.shape[:2]
        debug_canvas = image.copy() if debug_path is not None else None
        
        # Image Pre-processing with your ideal parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        edged = cv2.Canny(blurred, 45, 80) 
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Collect raw detections
        raw_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if (img_h * img_w * 0.002) < area < (img_h * img_w * 0.1):
                x, y, w, h = cv2.boundingRect(c)
                if 1.3 < (h/float(w)) < 1.6:
                    if not any(abs(x-bx)<20 and abs(y-by)<20 for bx,by,bw,bh in raw_boxes):
                        raw_boxes.append([x, y, w, h])

        if not raw_boxes: return {}, debug_canvas

        # NEW: Force Uniform Size using Median to ignore detection outliers
        uniform_w = int(np.median([b[2] for b in raw_boxes]))
        uniform_h = int(np.median([b[3] for b in raw_boxes]))

        # Infer Grid Coordinates using the uniform width/height
        xs = self._group_coords([b[0] for b in raw_boxes], uniform_w // 2)
        ys = self._group_coords([b[1] for b in raw_boxes], uniform_h // 2)

        final_results = {}
        slot_idx = 1
        for gy in ys:
            for gx in xs:
                # Apply standardized uniform size to every crop
                x_end = min(img_w, gx + uniform_w)
                y_end = min(img_h, gy + uniform_h)
                card_roi = image[max(0, gy):y_end, max(0, gx):x_end]
                
                # Check if the slot is empty
                if card_roi.size == 0 or np.max(cv2.meanStdDev(card_roi)[1]) < 18:
                    continue

                # Crop artwork based on fixed uniform dimensions
                ay1, ay2 = int(uniform_h * self.art_ratios['ay1']), int(uniform_h * self.art_ratios['ay2'])
                ax1, ax2 = int(uniform_w * self.art_ratios['ax1']), int(uniform_w * self.art_ratios['ax2'])
                art_roi = card_roi[ay1:ay2, ax1:ax2]

                if art_roi.size > 0:
                    match = self._get_best_match(art_roi)
                    final_results[f"Slot_{slot_idx:02d}"] = match
                    
                    # Save individual card debug images
                    if debug_path:
                        p_dist = match["p_dist"]
                        clean_name = "".join([c for c in match["name"] if c.isalnum() or c in ' _']).strip()
                        cv2.imwrite(str(debug_path / f"slot{slot_idx:02d}_{clean_name}_p{p_dist}.jpg"), art_roi)
                        
                        # Mark color on canvas with uniform box size
                        color = (0, 255, 0) if p_dist <= self.threshold else (0, 0, 255)
                        cv2.rectangle(debug_canvas, (gx, gy), (gx + uniform_w, gy + uniform_h), color, 2)
                    
                slot_idx += 1
        
        return final_results, debug_canvas

    def _get_best_match(self, art_roi):
        # Calculate Image Hash
        art_rgb = cv2.cvtColor(art_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(art_rgb).resize((128, 128), self.RESAMPLE_METHOD)
        curr_hash = imagehash.phash(pil_img, hash_size=16)
        
        # Calculate Color Histogram Score
        hsv = cv2.cvtColor(art_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        best_cand = None
        min_p = 999
        for cid, data in self.hash_db.items():
            p_dist = curr_hash - imagehash.hex_to_hash(data["phash"])
            if p_dist < min_p:
                min_p = p_dist
                c_score = cv2.compareHist(hist, np.array(data["hist"], dtype=np.float32).reshape(12, 8), cv2.HISTCMP_BHATTACHARYYA) if "hist" in data else 1.0
                info = self.id_db.get(cid, {"name": "Unknown", "type": "Unknown"})
                best_cand = {
                    "id": cid, "name": info["name"], "type": info["type"], 
                    "p_dist": int(p_dist), "c_score": round(float(c_score), 3)
                }
        return best_cand

    def _group_coords(self, coords, threshold):
        coords = sorted(coords)
        if not coords: return []
        groups = [[coords[0]]]
        for x in coords[1:]:
            if x - groups[-1][-1] < threshold: groups[-1].append(x)
            else: groups.append([x])
        return [int(np.mean(g)) for g in groups]

    def _format_output(self, results, user_id):
        output = {
            "user_id": user_id,
            "summary": {"total_matched": 0},
            "main_deck": {"Monster": {}, "Spell": {}, "Trap": {}},
            "extra_deck": {},
            "raw_results": results
        }
        for slot, data in results.items():
            if data['p_dist'] <= self.threshold:
                name, t = data['name'], data['type'].lower()
                # Categorize based on keywords
                is_extra = any(k in t for k in ["fusion", "synchro", "xyz", "link"])
                if is_extra:
                    output["extra_deck"][name] = output["extra_deck"].get(name, 0) + 1
                else:
                    cat = "Spell" if "spell" in t else ("Trap" if "trap" in t else "Monster")
                    output["main_deck"][cat][name] = output["main_deck"][cat].get(name, 0) + 1
                output["summary"]["total_matched"] += 1
        return output

# --- Example Usage ---
if __name__ == "__main__":
    recognizer = CardRecognizer(confidence_threshold=100)
    
    # Test with a local image file
    test_img = Path(__file__).parent.parent / 'data' / 'test_image4.png'
    if test_img.exists():
        result = recognizer.process_image(test_img, user_id="test_user", save_debug=True)
        print(f"[*] Recognition finished. Successfully matched {result['summary']['total_matched']} cards.")
    else:
        print(f"[!] Test image not found at {test_img}")