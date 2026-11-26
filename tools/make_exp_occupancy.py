import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 좌석 점유 예측 결과(pred_occupancy_A/B.json)와 
# GT(gt_occupancy.json)를 비교하여 
# 정확도 평가 지표를 exp_occupancy_A/B.json에 저장
# python tools/make_exp_occupancy.py --place_dir data/places/placeA --occ_method A
# python tools/make_exp_occupancy.py --place_dir data/places/placeA --occ_method B


# JSON 파일을 읽어올 때 확실하게 딕셔너리로 변환
with open('data/places/placeB/gt_occupancy.json', 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

# gt_data가 딕셔너리이고, 그 안의 "frames"가 리스트인지 확인

gt = gt_data.get("frames", [])

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def build_frame_map(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    m = {}
    for fr in frames:
        m[int(fr.get("frame", 0))] = fr
    return m



def compute_accuracy(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    """
    pred: pred_occupancy_{A|B}.json
      {frames:[{frame, seats:{id:bool}, tables:{id:bool}, ...}, ...]}
    gt: gt_occupancy.json
      {frames:[{frame, seats:{id:bool}, tables:{id:bool}}, ...]}
    """
    pred_map = build_frame_map(pred.get("frames", []))
    gt_map = build_frame_map(gt.get("frames", []))
    common_frames = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    if not common_frames:
        return {
            "frames_compared": 0,
            "overall_accuracy": 0.0,
            "seat_accuracy": 0.0,
            "table_accuracy": 0.0,
            "seat_n": 0,
            "table_n": 0,
            "overall_n": 0,
            "per_frame": []
        }

    seat_correct = seat_total = 0
    table_correct = table_total = 0
    per_frame = []

    for f in common_frames:
        p = pred_map[f]
        g = gt_map[f]
        p_seats: Dict[str, bool] = {k: bool(v) for k, v in (p.get("seats") or {}).items()}
        p_tables: Dict[str, bool] = {k: bool(v) for k, v in (p.get("tables") or {}).items()}
        g_seats: Dict[str, bool] = {k: bool(v) for k, v in (g.get("seats") or {}).items()}
        g_tables: Dict[str, bool] = {k: bool(v) for k, v in (g.get("tables") or {}).items()}

        # seat eval: GT에 있는 id만 평가(정답 기준)
        f_seat_correct = f_seat_total = 0
        for sid, gt_v in g_seats.items():
            f_seat_total += 1
            if bool(p_seats.get(sid, False)) == bool(gt_v):
                f_seat_correct += 1
        seat_correct += f_seat_correct
        seat_total += f_seat_total

        # table eval
        f_table_correct = f_table_total = 0
        for tid, gt_v in g_tables.items():
            f_table_total += 1
            if bool(p_tables.get(tid, False)) == bool(gt_v):
                f_table_correct += 1
        table_correct += f_table_correct
        table_total += f_table_total

        per_frame.append({
            "frame": f,
            "seat_accuracy": round(f_seat_correct / f_seat_total, 3) if f_seat_total else None,
            "table_accuracy": round(f_table_correct / f_table_total, 3) if f_table_total else None,
        })

    overall_correct = seat_correct + table_correct
    overall_total = seat_total + table_total

    return {
        "frames_compared": len(common_frames),
        "overall_accuracy": round(overall_correct / overall_total, 3) if overall_total else 0.0,
        "seat_accuracy": round(seat_correct / seat_total, 3) if seat_total else 0.0,
        "table_accuracy": round(table_correct / table_total, 3) if table_total else 0.0,
        "seat_n": seat_total,
        "table_n": table_total,
        "overall_n": overall_total,
        "per_frame": per_frame,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place_dir", required=True, help="e.g. data/places/placeA")
    ap.add_argument("--occ_method", required=True, choices=["A", "B"])
    args = ap.parse_args()

    place_dir = Path(args.place_dir)

    pred_path = place_dir / f"pred_occupancy_{args.occ_method}.json"
    gt_path = place_dir / "gt_occupancy.json"

    pred = load_json(pred_path)
    gt = load_json(gt_path)

    metrics = compute_accuracy(pred, gt)

    out = {
        "place": place_dir.name,
        "occ_method": args.occ_method,
        "pred_source": pred.get("source"),
        "frames_in_pred": len(pred.get("frames", [])),
        "frames_in_gt": len(gt.get("frames", [])),
        **metrics
    }

    out_path = place_dir / f"exp_occupancy_{args.occ_method}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
