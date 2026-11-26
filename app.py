from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
from flask import Flask, abort, jsonify, render_template, request, send_file
from threading import Lock



# =========================================================
# App / Paths
# =========================================================
app = Flask(__name__)

BASE_DIR = Path(__file__).parent
PLACES_DIR = BASE_DIR / "data" / "places"

#==============
# Player state
#==============
PLAYER_LOCK = Lock()
PLAYER_STATE = {}  # key=(place, occ_method) -> dict

def _player_key(place: str, occ_method: str) -> str:
    return f"{place}::{occ_method}"

def _ensure_player(place: str, occ_method: str) -> dict:
    """
    place+occ_method에 대한 player 상태를 초기화/보장
    상태:
      - playing: bool
      - idx: int (frame_nums의 index)
      - frame_nums: List[int]
    """
    key = _player_key(place, occ_method)
    st = PLAYER_STATE.get(key)
    if st is not None and st.get("frame_nums"):
        return st

    # pred_info의 frame_nums를 사용 (연속이 아닐 수 있으니 안전)
    pred = load_pred_occupancy(place, occ_method)
    frames = pred.get("frames", [])
    frame_nums = [int(fr.get("frame", i)) for i, fr in enumerate(frames)]

    st = {
        "place": place,
        "occ_method": occ_method,
        "playing": False,
        "idx": 0,
        "frame_nums": frame_nums,
    }
    PLAYER_STATE[key] = st
    return st

# =========================================================
# Helpers: basic IO / listing
# =========================================================
def list_places() -> List[str]:
    if not PLACES_DIR.exists():
        return []
    return sorted([p.name for p in PLACES_DIR.iterdir() if p.is_dir()])


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        abort(404, f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def load_roi(place: str) -> Dict[str, Any]:
    return read_json(PLACES_DIR / place / "roi.json")


def load_pred_occupancy(place: str, occ_method: str) -> Dict[str, Any]:
    # pred_occupancy_A.json or pred_occupancy_B.json
    return read_json(PLACES_DIR / place / f"pred_occupancy_{occ_method}.json")


def load_exp_yolo(place: str, variant: str) -> Dict[str, Any]:
    # exp_yolov8l.json (base), exp_yolov8l_ft.json (ft)
    if variant == "ft":
        return read_json(PLACES_DIR / place / "exp_yolov8l_ft.json")
    return read_json(PLACES_DIR / place / "exp_yolov8l.json")


def load_exp_occupancy(place: str, occ_method: str) -> Dict[str, Any]:
    # exp_occupancy_A.json / exp_occupancy_B.json
    return read_json(PLACES_DIR / place / f"exp_occupancy_{occ_method}.json")


# =========================================================
# Media frames cache
# Expected: data/places/<place>/media/frames/*.jpg|png
# Frame index -> sorted file list index (0-based)
# =========================================================
_FRAME_CACHE: Dict[str, Dict[str, Any]] = {}

def list_frame_files(place: str) -> List[Path]:
    frames_dir = PLACES_DIR / place / "media" / "frames"
    if not frames_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def frame_file_for_index(place: str, frame_idx: Optional[int]) -> Optional[Path]:
    frames_dir = PLACES_DIR / place / "media" / "frames"
    key = str(frames_dir)

    mtime = frames_dir.stat().st_mtime if frames_dir.exists() else None
    cached = _FRAME_CACHE.get(key)
    if cached is None or cached.get("mtime") != mtime:
        _FRAME_CACHE[key] = {"mtime": mtime, "files": list_frame_files(place)}
        cached = _FRAME_CACHE[key]

    files: List[Path] = cached.get("files", [])
    if not files:
        return None

    if frame_idx is None:
        return files[-1]

    i = int(frame_idx)
    if i < 0:
        return files[-1]
    if i >= len(files):
        return files[-1]
    return files[i]


# =========================================================
# Detection JSONL cache (fine-tuned default)
# Expected:
#   det_yolov8l.jsonl
#   det_yolov8l_ft.jsonl
# Each line:
#   {"frame": 9, "ts": 4.5, "people":[{x1,y1,x2,y2,conf}, ...], "infer_ms": ...}
# =========================================================
_DET_CACHE: Dict[str, Dict[str, Any]] = {}

def det_jsonl_path(place: str, variant: str) -> Path:
    if variant == "ft":
        return PLACES_DIR / place / "det_yolov8l_ft.jsonl"
    return PLACES_DIR / place / "det_yolov8l.jsonl"


def load_det_indexed(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(path)
    indexed: Dict[int, Dict[str, Any]] = {}
    for obj in rows:
        fr = obj.get("frame", None)
        if fr is None:
            continue
        try:
            fr_i = int(fr)
        except Exception:
            continue
        indexed[fr_i] = obj
    return indexed


def get_det_frame(place: str, variant: str, frame: Optional[int]) -> Optional[Dict[str, Any]]:
    path = det_jsonl_path(place, variant)
    key = str(path)
    mtime = path.stat().st_mtime if path.exists() else None
    cached = _DET_CACHE.get(key)

    if cached is None or cached.get("mtime") != mtime:
        _DET_CACHE[key] = {"mtime": mtime, "indexed": load_det_indexed(path)}
        cached = _DET_CACHE[key]

    indexed: Dict[int, Dict[str, Any]] = cached.get("indexed", {})
    if not indexed:
        return None

    if frame is None:
        last_fr = max(indexed.keys())
        return indexed[last_fr]
    return indexed.get(int(frame))


def det_people_as_boxes(det_fr: Optional[Dict[str, Any]], conf_thr: float = 0.0) -> List[Dict[str, Any]]:
    if not det_fr:
        return []
    out: List[Dict[str, Any]] = []
    for p in det_fr.get("people", []):
        c = float(p.get("conf", 1.0))
        if c < conf_thr:
            continue
        out.append({
            "x1": float(p.get("x1", 0)), "y1": float(p.get("y1", 0)),
            "x2": float(p.get("x2", 0)), "y2": float(p.get("y2", 0)),
            "conf": c
        })
    return out


# =========================================================
# Pred occupancy helpers
# Expected pred frame:
#   {"frame": 9, "ts": 4.5, "people_count": 2,
#    "seats": {"S01": false, ...}, "tables": {"T01": true, ...}}
# =========================================================
def pred_last_or_frame(pred: Dict[str, Any], frame: Optional[int]) -> Optional[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = pred.get("frames", [])
    if not frames:
        return None
    if frame is None:
        return frames[-1]
    for fr in frames:
        if int(fr.get("frame", -1)) == int(frame):
            return fr
    return None


def compute_rates_from_pred_frame(
    roi: Dict[str, Any],
    fr: Dict[str, Any],
    people_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dashboard KPI는 '마지막 프레임의 pred_occupancy'를 기준으로 계산.
    - seats/tables 점유율은 pred dict를 1순위로 사용 (ID mismatch 방지)
    - ROI는 표시/정렬(좌석 리스트 순서) 목적
    """

    seats_map: Dict[str, bool] = {k: bool(v) for k, v in (fr.get("seats") or {}).items()}
    tables_map: Dict[str, bool] = {k: bool(v) for k, v in (fr.get("tables") or {}).items()}

    # ROI id lists (for stable ordering / fallback)
    seat_ids_roi = [s.get("seat_id") for s in roi.get("seats", []) if s.get("seat_id")]
    table_ids_roi = [t.get("table_id") for t in roi.get("tables", []) if t.get("table_id")]

    # --- Seat rate: prefer ROI ordering but count from seats_map ---
    if seats_map:
        total_seats = len(seats_map)
        occ_seats = sum(1 for v in seats_map.values() if v)
    else:
        total_seats = len(seat_ids_roi)
        occ_seats = 0

    seat_rate = round((occ_seats / total_seats) * 100.0, 1) if total_seats else 0.0

    # Prefer pred tables_map directly.
    if tables_map:
        total_tables = len(tables_map)
        occ_tables = sum(1 for v in tables_map.values() if v)
    else:
        # fallback: ROI-based
        total_tables = len(table_ids_roi)
        occ_tables = 0

    table_rate = round((occ_tables / total_tables) * 100.0, 1) if total_tables else 0.0

    people = int(people_override) if people_override is not None else int(fr.get("people_count", 0))

    # Simple crowd level score (adjustable)
    score = 0.6 * (seat_rate / 100.0) + 0.2 * (table_rate / 100.0) + 0.2 * min(1.0, people / 20.0)
    if score < 0.4:
        level = "LOW"
    elif score < 0.7:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "people_count": people,
        "seat_rate": seat_rate,
        "table_rate": table_rate,
        "crowd_level": level,

        # For UI
        "seats_map": seats_map,
        "tables_map": tables_map,
        "seat_ids_order": seat_ids_roi if seat_ids_roi else list(seats_map.keys()),
        "table_ids_order": table_ids_roi if table_ids_roi else list(tables_map.keys()),
    }


# =========================================================
# Pages
# =========================================================
@app.route("/")
def page_dashboard():
    place = request.args.get("place", "placeA")
    return render_template("dashboard.html", place=place, places=list_places())


@app.route("/occupancy")
def page_occupancy():
    place = request.args.get("place", "placeA")
    return render_template("occupancy.html", place=place, places=list_places())


@app.route("/congestion")
def page_congestion():
    place = request.args.get("place", "placeA")
    return render_template("congestion.html", place=place, places=list_places())


@app.route("/experiments")
def page_experiments():
    place = request.args.get("place", "placeA")
    return render_template("experiments.html", place=place, places=list_places())

@app.route("/about")
def page_about():
    place = request.args.get("place", "placeA")
    return render_template("about.html", place=place, places=list_places())

# =========================================================
# APIs
# =========================================================
@app.route("/api/place_info")
def api_place_info():
    place = request.args.get("place", "placeA")
    roi = load_roi(place)
    return jsonify({
        "place": place,
        "image_w": roi.get("image_w"),
        "image_h": roi.get("image_h"),
        "num_seats": len(roi.get("seats", [])),
        "num_tables": len(roi.get("tables", [])),
        "roi": roi
    })


@app.route("/api/frame_image")
def api_frame_image():
    """
    Serve a frame image for the selected place.
    query:
      place=<placeA>
      frame=<int> (optional; default: last available)
    """
    place = request.args.get("place", "placeA")
    frame = request.args.get("frame")
    frame_i = int(frame) if frame is not None else None

    p = frame_file_for_index(place, frame_i)
    if not p or not p.exists():
        abort(404, f"frame image not found for place={place}, frame={frame_i}")

    return send_file(str(p))

@app.route("/api/player/state")
def api_player_state():
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")

    with PLAYER_LOCK:
        st = _ensure_player(place, occ_method)
        n = len(st["frame_nums"])
        idx = max(0, min(st["idx"], max(0, n-1)))
        st["idx"] = idx
        frame = st["frame_nums"][idx] if n else None

        return jsonify({
            "place": place,
            "occ_method": occ_method,
            "playing": bool(st["playing"]),
            "idx": idx,
            "n_frames": n,
            "frame": frame,
        })

@app.route("/api/player/start", methods=["POST"])
def api_player_start():
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")

    with PLAYER_LOCK:
        st = _ensure_player(place, occ_method)
        st["playing"] = True
    return jsonify({"ok": True})

@app.route("/api/player/stop", methods=["POST"])
def api_player_stop():
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")

    with PLAYER_LOCK:
        st = _ensure_player(place, occ_method)
        st["playing"] = False
    return jsonify({"ok": True})

@app.route("/api/player/reset", methods=["POST"])
def api_player_reset():
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")

    with PLAYER_LOCK:
        st = _ensure_player(place, occ_method)
        st["playing"] = False
        st["idx"] = 0
    return jsonify({"ok": True})

@app.route("/api/player/step", methods=["POST"])
def api_player_step():
    """
    playing==True일 때 1 step 전진. 끝까지 가면 stop.
    """
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")

    with PLAYER_LOCK:
        st = _ensure_player(place, occ_method)
        n = len(st["frame_nums"])
        if not n:
            return jsonify({"ok": True, "frame": None, "playing": False})

        if not st["playing"]:
            # 정지 상태면 그대로 반환
            idx = max(0, min(st["idx"], n-1))
            st["idx"] = idx
            return jsonify({"ok": True, "frame": st["frame_nums"][idx], "playing": False})

        # 재생 중이면 한 칸 전진
        idx = st["idx"] + 1
        if idx >= n:
            st["idx"] = n-1
            st["playing"] = False
        else:
            st["idx"] = idx

        return jsonify({
            "ok": True,
            "frame": st["frame_nums"][st["idx"]],
            "playing": bool(st["playing"]),
            "idx": st["idx"],
            "n_frames": n,
        })

@app.route("/api/pred_info")
def api_pred_info():
    """
    pred_occupancy_{A|B}.json의 프레임 메타 제공
    query:
      place
      occ_method=A|B
    """
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")
    pred = load_pred_occupancy(place, occ_method)
    frames = pred.get("frames", [])
    if not frames:
        return jsonify({"place": place, "occ_method": occ_method, "n_frames": 0, "frame_min": None, "frame_max": None})

    # pred 내 frame 값이 0..N-1이 아닐 수도 있어서 실제 frame 번호 범위를 제공
    frame_nums = [int(fr.get("frame", i)) for i, fr in enumerate(frames)]
    return jsonify({
        "place": place,
        "occ_method": occ_method,
        "n_frames": len(frames),
        "frame_min": min(frame_nums),
        "frame_max": max(frame_nums),
        "frame_nums": frame_nums  # UI에서 정확히 순차 재생하려면 이게 가장 안전
    })

@app.route("/api/detections")
def api_detections():
    """
    Return YOLO detections from det_yolov8l*.jsonl for a given frame.
    query:
      place=<placeA>
      variant=ft|base (default: ft)
      frame=<int> (optional; default: last)
      conf=<float> (optional; default: 0.25)
    """
    place = request.args.get("place", "placeA")
    variant = request.args.get("variant", "ft")
    frame = request.args.get("frame")
    frame_i = int(frame) if frame is not None else None
    conf = float(request.args.get("conf", 0.25))

    det_fr = get_det_frame(place, variant, frame_i)
    det_people = det_people_as_boxes(det_fr, conf_thr=conf)

    return jsonify({
        "place": place,
        "variant": variant,
        "frame": int(det_fr.get("frame", -1)) if det_fr else frame_i,
        "timestamp": float(det_fr.get("ts", 0.0)) if det_fr else None,
        "people_count": len(det_people),
        "detections": det_people
    })


@app.route("/api/seats")
def api_seats():
    """
    pred_occupancy_{A|B}.json 기반으로 “현재(기본: 마지막) 프레임” 점유 상태 반환.
    Dashboard/Occupancy가 이 API를 사용해:
      - ROI 좌석/테이블에 점유 색상 반영
      - det_yolov8l_ft.jsonl bbox 오버레이
      - 최신 프레임 번호로 이미지 표시
    query:
      place
      occ_method=A|B (default B)
      frame=<int> optional (default last)
      det_variant=ft|base (default ft)
      conf=<float> optional (default 0.25) : bbox 표시용
    """
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")
    frame = request.args.get("frame")
    frame_i = int(frame) if frame is not None else None

    det_variant = request.args.get("det_variant", "ft")
    conf = float(request.args.get("conf", 0.25))

    roi = load_roi(place)
    pred = load_pred_occupancy(place, occ_method)
    pred_fr = pred_last_or_frame(pred, frame_i)
    if not pred_fr:
        return jsonify({
            "place": place, "occ_method": occ_method, "frame": frame_i,
            "seats": [], "tables": [], "detections": [],
            "note": "no pred frames"
        })

    fr_no = int(pred_fr.get("frame", -1))
    det_fr = get_det_frame(place, det_variant, fr_no)
    det_people = det_people_as_boxes(det_fr, conf_thr=conf)

    # people_count는 ft det 기반으로 보정(있으면)
    people_cnt = len(det_people) if det_fr else int(pred_fr.get("people_count", 0))

    met = compute_rates_from_pred_frame(roi, pred_fr, people_override=people_cnt)

    seats = [{"seat_id": sid, "occupied": bool(met["seats_map"].get(sid, False))}
             for sid in met["seat_ids_order"]]
    tables = [{"table_id": tid, "occupied": bool(met["tables_map"].get(tid, False))}
              for tid in met["table_ids_order"]]

    return jsonify({
        "place": place,
        "occ_method": occ_method,
        "frame": fr_no,
        "timestamp": float(pred_fr.get("ts", 0.0)),
        "people_count": met["people_count"],
        "seat_occupancy_rate": met["seat_rate"],
        "table_occupancy_rate": met["table_rate"],
        "crowd_level": met["crowd_level"],
        "detections": det_people,
        "seats": seats,
        "tables": tables,
        "source": f"pred_occupancy_{occ_method} + det_yolov8l_{det_variant}.jsonl"
    })


@app.route("/api/metrics")
def api_metrics():
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")
    frame_req = request.args.get("frame", None)  

    pred = load_pred_occupancy(place, occ_method)  # pred_occupancy_{A|B}.json
    frames = pred.get("frames", [])
    if not frames:
        return jsonify({
            "place": place, "occ_method": occ_method,
            "frame": None, "timestamp": None,
            "people_count": 0,
            "seat_occupancy_rate": 0,
            "table_occupancy_rate": 0,
            "crowd_level": "LOW",
            "source": f"pred_occupancy_{occ_method}.json (empty)"
        })

    # frame 선택: 요청 frame이 있으면 그 frame을, 없으면 마지막 프레임
    if frame_req is None:
        fr_obj = frames[-1]
    else:
        frn = int(frame_req)
        # pred frames는 list이지만 frame 번호가 index와 다를 수 있어 탐색
        fr_obj = None
        for item in frames:
            if int(item.get("frame", -1)) == frn:
                fr_obj = item
                break
        if fr_obj is None:
            # 못 찾으면 안전하게 마지막 프레임
            fr_obj = frames[-1]

    seats = fr_obj.get("seats", {}) or {}
    tables = fr_obj.get("tables", {}) or {}

    total_seats = len(seats)
    occ_seats = sum(1 for v in seats.values() if bool(v))
    seat_rate = round((occ_seats / total_seats) * 100.0, 1) if total_seats else 0.0

    total_tables = len(tables)
    occ_tables = sum(1 for v in tables.values() if bool(v))
    table_rate = round((occ_tables / total_tables) * 100.0, 1) if total_tables else 0.0

    people = int(fr_obj.get("people_count", 0))
    ts = fr_obj.get("ts", None)
    frame_no = int(fr_obj.get("frame", 0))

    # 혼잡도 레벨(기존 간단 결합 유지)
    crowd_score = 0.7*(seat_rate/100.0) + 0.3*min(1.0, people/20.0)
    if crowd_score < 0.4: level = "LOW"
    elif crowd_score < 0.7: level = "MEDIUM"
    else: level = "HIGH"

    return jsonify({
        "place": place,
        "occ_method": occ_method,
        "frame": frame_no,
        "timestamp": ts,
        "people_count": people,
        "seat_occupancy_rate": seat_rate,
        "table_occupancy_rate": table_rate,
        "crowd_level": level,
        "source": f"pred_occupancy_{occ_method}.json"
    })


@app.route("/api/history")
def api_history():
    """
    혼잡도/점유율 time-series 그래프용:
    - pred_occupancy의 최근 n 프레임을 반환
    query:
      place
      occ_method=A|B (default B)
      n=120 default
    """
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")
    n = int(request.args.get("n", 120))

    roi = load_roi(place)
    pred = load_pred_occupancy(place, occ_method)
    frames = pred.get("frames", [])
    if not frames:
        return jsonify({"place": place, "occ_method": occ_method, "points": []})

    tail = frames[-n:] if n > 0 else frames
    points = []
    for fr in tail:
        met = compute_rates_from_pred_frame(roi, fr)
        points.append({
            "t": float(fr.get("ts", 0.0)),
            "frame": int(fr.get("frame", -1)),
            "people": int(met["people_count"]),
            "seat_rate": float(met["seat_rate"]),
            "table_rate": float(met["table_rate"]),
            "level": met["crowd_level"]
        })

    return jsonify({"place": place, "occ_method": occ_method, "points": points})


@app.route("/api/heatmap_roi")
def api_heatmap_roi():
    """
    ROI별 점유 빈도(0~1) 계산(최근 frames window)
    query:
      place
      occ_method
      frames=60 default
      target=all|seats|tables
    """
    place = request.args.get("place", "placeA")
    occ_method = request.args.get("occ_method", "B")
    frames_n = int(request.args.get("frames", 60))
    target = request.args.get("target", "all")

    roi = load_roi(place)
    pred = load_pred_occupancy(place, occ_method)
    frames = pred.get("frames", [])
    if not frames:
        return jsonify({"place": place, "occ_method": occ_method, "target": target, "values": {}, "frames_used": 0})

    window = frames[-frames_n:] if frames_n > 0 else frames

    seat_ids = [s.get("seat_id") for s in roi.get("seats", []) if s.get("seat_id")] if target in ("seats", "all") else []
    table_ids = [t.get("table_id") for t in roi.get("tables", []) if t.get("table_id")] if target in ("tables", "all") else []

    counts: Dict[str, int] = {rid: 0 for rid in (seat_ids + table_ids)}
    for fr in window:
        s_map = fr.get("seats") or {}
        t_map = fr.get("tables") or {}
        for sid in seat_ids:
            if bool(s_map.get(sid, False)):
                counts[sid] += 1
        for tid in table_ids:
            if bool(t_map.get(tid, False)):
                counts[tid] += 1

    n_used = len(window)
    values = {rid: round(counts[rid] / n_used, 3) if n_used else 0.0 for rid in counts.keys()}

    return jsonify({
        "place": place,
        "occ_method": occ_method,
        "target": target,
        "frames_used": n_used,
        "values": values,
        "source": "pred_occupancy_window"
    })


# -------------------------
# Experiments APIs
# -------------------------
@app.route("/api/experiments/detection")
def api_exp_detection():
    """
    Detection summary only (base vs ft).
    Reads exp_yolov8l.json and exp_yolov8l_ft.json
    """
    place = request.args.get("place", "placeA")
    base = load_exp_yolo(place, "base")
    ft = load_exp_yolo(place, "ft")

    def pack(x: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": x.get("model"),
            "precision": x.get("precision"),
            "recall": x.get("recall"),
            "tp": x.get("tp"),
            "fp": x.get("fp"),
            "fn": x.get("fn"),
            "avg_infer_ms": x.get("avg_infer_ms"),
            "avgFPS": x.get("avgFPS"),
            "frames_compared": x.get("frames_compared"),
        }

    return jsonify({"place": place, "base": pack(base), "ft": pack(ft)})


@app.route("/api/experiments/detection_detail")
def api_exp_detection_detail():
    """
    Detection detail (includes per_frame) for plotting.
    """
    place = request.args.get("place", "placeA")
    base = load_exp_yolo(place, "base")
    ft = load_exp_yolo(place, "ft")
    return jsonify({"place": place, "base": base, "ft": ft})


@app.route("/api/experiments/occupancy")
def api_exp_occupancy():
    """
    Occupancy summary only (A vs B).
    Reads exp_occupancy_A.json and exp_occupancy_B.json
    """
    place = request.args.get("place", "placeA")
    exp_A = load_exp_occupancy(place, "A") if (PLACES_DIR / place / "exp_occupancy_A.json").exists() else {}
    exp_B = load_exp_occupancy(place, "B") if (PLACES_DIR / place / "exp_occupancy_B.json").exists() else {}

    def pack(x: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "overall_accuracy": x.get("overall_accuracy"),
            "seat_accuracy": x.get("seat_accuracy"),
            "table_accuracy": x.get("table_accuracy"),
            "frames_compared": x.get("frames_compared"),
            "overall_n": x.get("overall_n"),
            "pred_source": x.get("pred_source"),
        }

    return jsonify({"place": place, "logicA": pack(exp_A), "logicB": pack(exp_B)})


@app.route("/api/experiments/occupancy_detail")
def api_exp_occupancy_detail():
    """
    Occupancy detail (includes per_frame) for plotting.
    """
    place = request.args.get("place", "placeA")
    exp_A = load_exp_occupancy(place, "A")
    exp_B = load_exp_occupancy(place, "B")
    return jsonify({"place": place, "logicA": exp_A, "logicB": exp_B})


if __name__ == "__main__":
    app.run(debug=True)
