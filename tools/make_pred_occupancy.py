import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

# roi.json의 좌석/테이블 영역과
# gt_people.jsonl 또는 det_yolov8l_ft.jsonl의 사람 박스들을 이용하여
# 좌석/테이블 점유 예측 결과(pred_occupancy_A/B.json)를 생성
# python tools/make_pred_occupancy.py --place_dir data/places/placeA --source det_yolov8l_ft.jsonl --occ_method A --conf 0.25
# python tools/make_pred_occupancy.py --place_dir data/places/placeA --source det_yolov8l_ft.jsonl --occ_method B --conf 0.25
# python tools/make_pred_occupancy.py --place_dir data/places/placeA --source det_yolov8l_ft.jsonl --occ_method B --conf 0.25

# -------------------------
# Geometry / logic (same as server)
# -------------------------
@dataclass(frozen=True)
class Box:
    x1: float; y1: float; x2: float; y2: float
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = Box(ix1, iy1, ix2, iy2).area()
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0

def foot_point(pb: Box, alpha: float = 0.12) -> Tuple[float, float]:
    xc = (pb.x1 + pb.x2) / 2.0
    h = max(1e-6, pb.y2 - pb.y1)
    y = pb.y2 - alpha * h
    return xc, y

def point_in_polygon(pt: Tuple[float, float], poly: List[List[float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < xinters:
                inside = not inside
    return inside

def roi_contains_point(roi_item: Dict[str, Any], pt: Tuple[float, float]) -> bool:
    typ = roi_item.get("type", "rect")
    if typ == "polygon":
        return point_in_polygon(pt, roi_item["points"])
    r = roi_item["roi"]
    return (r["x1"] <= pt[0] <= r["x2"]) and (r["y1"] <= pt[1] <= r["y2"])

def roi_box_if_rect(roi_item: Dict[str, Any]) -> Optional[Box]:
    if roi_item.get("type", "rect") != "rect":
        return None
    r = roi_item["roi"]
    return Box(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))

def polygon_bbox(poly: List[List[float]]) -> Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return Box(min(xs), min(ys), max(xs), max(ys))

def occupancy_A_iou(people: List[Box], roi_list: List[Dict[str, Any]], iou_thr: float = 0.08) -> Dict[str, bool]:
    occ: Dict[str, bool] = {}
    for item in roi_list:
        rid = item.get("seat_id") or item.get("table_id") or item.get("roi_id")
        occ[rid] = False
        if item.get("type", "rect") == "rect":
            rb = roi_box_if_rect(item)
        else:
            rb = polygon_bbox(item["points"])
        for pb in people:
            if iou(pb, rb) >= iou_thr:
                occ[rid] = True
                break
    return occ

def occupancy_B_footpoint(people: List[Box], roi_list: List[Dict[str, Any]], alpha: float = 0.12) -> Dict[str, bool]:
    occ: Dict[str, bool] = {}
    for item in roi_list:
        rid = item.get("seat_id") or item.get("table_id") or item.get("roi_id")
        occ[rid] = False
        for pb in people:
            if roi_contains_point(item, foot_point(pb, alpha=alpha)):
                occ[rid] = True
                break
    return occ

def parse_people_from_people_list(people_list: List[Dict[str, Any]], conf_thr: float = 0.25) -> List[Box]: #conf수정
    out: List[Box] = []
    for p in people_list:
        if "conf" in p and float(p.get("conf", 1.0)) < conf_thr:
            continue
        out.append(Box(float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])))
    return out

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place_dir", required=True, help="e.g. data/places/placeA")
    ap.add_argument("--source", required=True, help="e.g. gt_people.jsonl or det_yolov8l_ft.jsonl")
    ap.add_argument("--occ_method", required=True, choices=["A", "B"])
    ap.add_argument("--conf", type=float, default=0.25, help="used only if source has conf fields")
    ap.add_argument("--fps_assumed", type=float, default=2.0, help="for ts if missing (default: 0.5s step)")
    args = ap.parse_args()

    place_dir = Path(args.place_dir)
    roi = json.loads((place_dir / "roi.json").read_text(encoding="utf-8"))
    src_path = place_dir / args.source

    frames = read_jsonl(src_path)
    out_frames = []
    dt = 1.0 / max(1e-9, args.fps_assumed)

    for fr in frames:
        idx = int(fr.get("frame", 0))
        ts = float(fr["ts"]) if "ts" in fr else idx * dt
        people = parse_people_from_people_list(fr.get("people", []), conf_thr=args.conf)

        if args.occ_method == "A":
            s_occ = occupancy_A_iou(people, roi.get("seats", []))
            t_occ = occupancy_A_iou(people, roi.get("tables", []))
        else:
            s_occ = occupancy_B_footpoint(people, roi.get("seats", []))
            t_occ = occupancy_B_footpoint(people, roi.get("tables", []))

        out_frames.append({
            "frame": idx,
            "ts": ts,
            "people_count": len(people),
            "seats": {k: bool(v) for k, v in s_occ.items()},
            "tables": {k: bool(v) for k, v in t_occ.items()},
        })

    out_obj = {
        "place": place_dir.name,
        "occ_method": args.occ_method,
        "source": args.source,
        "fps_assumed": args.fps_assumed,
        "frames": out_frames
    }

    out_path = place_dir / f"pred_occupancy_{args.occ_method}.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path} (frames={len(out_frames)})")

if __name__ == "__main__":
    main()
