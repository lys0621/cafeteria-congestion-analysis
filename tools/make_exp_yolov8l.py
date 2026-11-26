import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import math
import statistics
import cv2  # [추가] OpenCV
import numpy as np # [추가] Numpy

# python tools/make_exp_yolov8l.py --place placeA --gt data/places/placeA/gt_people.jsonl --det data/places/placeA/det_yolov8l.jsonl --out data/places/placeA/exp_yolov8l.json --model_tag yolov8l --conf 0.25 --iou 0.5 --roi roi.json
# -------------------------
# Geometry
# -------------------------

class Box:
    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, x1, y1, x2, y2):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


# -------------------------
# ROI Helper [추가됨]
# -------------------------
def load_roi_mask(roi_path: Path) -> Optional[np.ndarray]:
    """
    roi.json을 읽어서 Seat와 Table 영역이 255로 칠해진 Binary Mask를 반환합니다.
    """
    if not roi_path.exists():
        print(f"[Warning] ROI file not found: {roi_path}")
        return None

    with open(roi_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 크기 가져오기 (기본값 설정 또는 JSON에서 읽기)
    h = data.get("image_h", 816)
    w = data.get("image_w", 720)
    
    # 빈 마스크 생성 (검은색)
    mask = np.zeros((h, w), dtype=np.uint8)

    polygons = []
    # Seats와 Tables 모두 가져오기
    items = data.get("seats", []) + data.get("tables", [])
    
    for item in items:
        if item.get("type") == "polygon":
            points = item.get("points", [])
            if points:
                # 좌표를 정수형 numpy 배열로 변환
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                polygons.append(pts)
    
    # 마스크에 폴리곤 그리기 (흰색 채우기)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
        
    return mask

def check_roi_overlap(box: Box, mask: np.ndarray) -> bool:
    """
    박스 영역과 마스크(ROI)가 한 픽셀이라도 겹치는지 확인합니다.
    """
    if mask is None:
        return True # 마스크가 없으면 필터링하지 않음

    h, w = mask.shape
    
    # 좌표 정수 변환 및 이미지 범위 내로 클리핑
    x1 = max(0, int(box.x1))
    y1 = max(0, int(box.y1))
    x2 = min(w, int(box.x2))
    y2 = min(h, int(box.y2))

    # 유효하지 않은 박스 (크기가 0 이하)
    if x2 <= x1 or y2 <= y1:
        return False

    # 마스크에서 해당 박스 영역만 잘라냄 (ROI Slice)
    roi_crop = mask[y1:y2, x1:x2]

    # 흰색 픽셀(255)이 하나라도 있으면 True (겹침)
    return cv2.countNonZero(roi_crop) > 0


# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    results = []
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, idx = decoder.raw_decode(text, idx)
            results.append(obj)
        except json.JSONDecodeError:
            break
        
    return results

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(statistics.mean(xs))


# -------------------------
# Matching / evaluation
# -------------------------
def greedy_match_counts(
    gt_boxes: List[Box],
    det_boxes: List[Box],
    iou_thr: float
) -> Tuple[int, int, int]:
    used_gt = [False] * len(gt_boxes)
    tp = 0

    for db in det_boxes:
        best_j = -1
        best_iou = 0.0
        for j, gb in enumerate(gt_boxes):
            if used_gt[j]:
                continue
            v = iou(db, gb)
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            tp += 1
            used_gt[best_j] = True

    fp = len(det_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def pr_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


# -------------------------
# Main builder
# -------------------------
def make_detection_exp_summary(
    gt_path: Path,
    det_path: Path,
    out_path: Path,
    roi_path: Optional[Path], # [추가] ROI 경로
    place: str,
    model_tag: str,
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
) -> None:
    
    # 0. ROI 마스크 로드 (한 번만 수행) [추가]
    roi_mask = None
    if roi_path:
        roi_mask = load_roi_mask(roi_path)
        if roi_mask is not None:
            print(f"[Info] Loaded ROI mask from {roi_path}")

    # 1. 데이터 로드
    gt_raw = load_jsonl(gt_path)
    if len(gt_raw) == 1 and "frames" in gt_raw[0]:
        gt_rows = gt_raw[0]["frames"]
    else:
        gt_rows = gt_raw

    det_raw = load_jsonl(det_path)
    if len(det_raw) == 1 and "frames" in det_raw[0]:
        det_rows = det_raw[0]["frames"]
    else:
        det_rows = det_raw

    gt_map = {int(r.get("frame", 0)): r for r in gt_rows}
    det_map = {int(r.get("frame", 0)): r for r in det_rows}

    frames_gt = sorted(gt_map.keys())
    frames_det = sorted(det_map.keys())
    frames = sorted(set(frames_gt) & set(frames_det))

    per_frame: List[Dict[str, Any]] = []

    TP = FP = FN = 0
    infer_ms_list: List[float] = []

    # frame-wise evaluation
    for fr in frames:
        gt_boxes = [
            Box(p["x1"], p["y1"], p["x2"], p["y2"])
            for p in gt_map[fr].get("people", [])
        ]
        det_boxes = []
        for p in det_map[fr].get("people", []):
            # 1. 확신도(conf) 체크
            if float(p.get("conf", 1.0)) < conf_thr:
                continue
                
            # 2. 천장 박스 무시
            if p["y1"] < 10: 
                continue
            
            # [추가] 3. ROI(좌석/테이블) 겹침 체크
            # 박스 객체 생성
            current_box = Box(p["x1"], p["y1"], p["x2"], p["y2"])
            
            # ROI 마스크가 있다면, 겹치지 않는 경우 continue (제외)
            if roi_mask is not None:
                if not check_roi_overlap(current_box, roi_mask):
                    continue

            # 통과한 것만 담기
            det_boxes.append(current_box)

        tp, fp, fn = greedy_match_counts(gt_boxes, det_boxes, iou_thr=iou_thr)
        TP += tp
        FP += fp
        FN += fn

        p, r = pr_from_counts(tp, fp, fn)

        infer_ms = det_map[fr].get("infer_ms")
        if infer_ms is not None:
            try:
                infer_ms_list.append(float(infer_ms))
            except Exception:
                pass

        per_frame.append({
            "frame": fr,
            "gt_count": len(gt_boxes),
            "det_count": len(det_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "infer_ms": infer_ms
        })

    # overall metrics
    overall_p, overall_r = pr_from_counts(TP, FP, FN)

    avg_infer_ms = safe_mean(infer_ms_list)
    avg_fps = None
    if avg_infer_ms and avg_infer_ms > 0:
        avg_fps = round(1000.0 / avg_infer_ms, 2)

    out = {
        "place": place,
        "model": model_tag,
        "gt_source": gt_path.name,
        "det_source": det_path.name,
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "roi_filter": roi_path.name if roi_path else None, # 정보 기록

        "frames_in_gt": len(frames_gt),
        "frames_in_det": len(frames_det),
        "frames_compared": len(frames),

        "tp": TP,
        "fp": FP,
        "fn": FN,
        "precision": round(overall_p, 3),
        "recall": round(overall_r, 3),

        "avg_infer_ms": (round(avg_infer_ms, 2) if avg_infer_ms is not None else None),
        "avgFPS": avg_fps,

        "per_frame": per_frame
    }

    save_json(out_path, out)
    print(f"[OK] wrote {out_path} (frames={len(frames)})")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--place", required=True, help="placeA / placeB ...")
    ap.add_argument("--gt", required=True, help="path to gt_people.jsonl")
    ap.add_argument("--det", required=True, help="path to det_yolov8l*.jsonl")
    ap.add_argument("--out", required=True, help="path to exp_yolov8l*.json")
    ap.add_argument("--roi", required=False, help="path to roi.json for filtering") # [추가]
    ap.add_argument("--model_tag", default="yolov8l", help='e.g. "yolov8l" or "yolov8l_ft"')
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    roi_p = Path(args.roi) if args.roi else None

    make_detection_exp_summary(
        gt_path=Path(args.gt),
        det_path=Path(args.det),
        out_path=Path(args.out),
        roi_path=roi_p,
        place=args.place,
        model_tag=args.model_tag,
        conf_thr=args.conf,
        iou_thr=args.iou,
    )