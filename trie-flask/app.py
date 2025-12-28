import os
import re
import io
import time
import gc
import base64
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, send_file, flash


# =========================================================
# CONFIG PATH DATASET
# =========================================================
BASE_DIR = os.path.dirname(__file__)
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "amazon_products.csv")
REQUIRED_COL = "product_name"

# Preview options
PREVIEW_CHOICES = [20, 50, 100]
DEFAULT_PREVIEW_N = 20


app = Flask(__name__)
app.secret_key = "trie-autocomplete-aka-secret"


# =========================
# PREPROCESS
# =========================
def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return sorted(set([x for x in out if x > 0]))


def parse_prefixes(multiline: str) -> List[str]:
    lines = [x.strip() for x in multiline.splitlines() if x.strip()]
    cleaned = [preprocess_text(x) for x in lines]
    return [x for x in cleaned if x]


# =========================
# TIMING (median + warmup)
# =========================
def timed_call_ms(func, repeats: int = 7, warmup: int = 2) -> Tuple[float, float]:
    """
    Jalankan func beberapa kali. Return (median_ms, stdev_ms).
    - warmup: pemanasan (tidak dicatat)
    - repeats: jumlah pengukuran dicatat (disarankan ganjil: 5/7/9)
    """
    for _ in range(max(0, warmup)):
        func()

    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000_000.0)

    med = statistics.median(times)
    sd = statistics.pstdev(times) if len(times) > 1 else 0.0
    return med, sd


# =========================
# TRIE
# =========================
@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _find_prefix_node(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    # ITERATIVE DFS
    def suggest_iterative(self, prefix: str, limit: int = 10) -> List[str]:
        node = self._find_prefix_node(prefix)
        if node is None:
            return []
        res: List[str] = []
        stack: List[Tuple[TrieNode, str]] = [(node, "")]
        while stack and len(res) < limit:
            curr, path = stack.pop()
            if curr.is_end:
                res.append(prefix + path)
                if len(res) >= limit:
                    break
            for ch, nxt in curr.children.items():
                stack.append((nxt, path + ch))
        return res

    # RECURSIVE DFS (bounded)
    def suggest_recursive(self, prefix: str, limit: int = 10, max_depth: int = 200) -> List[str]:
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        res: List[str] = []

        def dfs(curr: TrieNode, path: str, depth: int):
            if depth > max_depth or len(res) >= limit:
                return
            if curr.is_end:
                res.append(prefix + path)
                if len(res) >= limit:
                    return
            for ch, nxt in curr.children.items():
                dfs(nxt, path + ch, depth + 1)
                if len(res) >= limit:
                    return

        dfs(node, "", 0)
        return res


def build_trie(words: List[str]) -> Trie:
    t = Trie()
    for w in words:
        if w:
            t.insert(w)
    return t


# =========================
# TEMPLATE FILTER
# =========================
@app.template_filter("b64")
def b64_filter(data: bytes) -> str:
    if not data:
        return ""
    return base64.b64encode(data).decode("utf-8")


# =========================
# PLOT HELPERS
# =========================
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_line(x, y, xlabel, ylabel, title, points=True, smooth_y=None, label_raw="raw", label_trend="trend") -> bytes:
    fig = plt.figure()
    if points:
        plt.plot(x, y, "o", alpha=0.55, label=label_raw)
    if smooth_y is not None:
        plt.plot(x, smooth_y, "-", linewidth=2.2, label=label_trend)
    else:
        plt.plot(x, y, "-", linewidth=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if points or smooth_y is not None:
        plt.legend()
    return fig_to_png_bytes(fig)


def plot_overlay(x, y1_s, y2_s, xlabel, ylabel, title, label1, label2) -> bytes:
    fig = plt.figure()
    plt.plot(x, y1_s, "-", linewidth=2.2, label=label1)
    plt.plot(x, y2_s, "-", linewidth=2.2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return fig_to_png_bytes(fig)


# =========================
# DATASET IO + PREVIEW
# =========================
def get_dataset_info(csv_path: str) -> Tuple[bool, str, int]:
    if not csv_path:
        return False, "Path dataset kosong.", 0
    if not os.path.exists(csv_path):
        return False, f"File tidak ditemukan: {csv_path}", 0
    try:
        df = pd.read_csv(csv_path, usecols=[REQUIRED_COL])
        return True, "Dataset siap.", len(df)
    except Exception as e:
        return False, f"Gagal baca CSV/kolom '{REQUIRED_COL}': {e}", 0


def load_preview(csv_path: str, max_preview: int) -> Tuple[List[Dict], List[str]]:
    df = pd.read_csv(csv_path, usecols=[REQUIRED_COL]).head(max_preview).copy()
    df["raw"] = df[REQUIRED_COL].astype(str)
    df["cleaned"] = df["raw"].apply(preprocess_text)
    preview = df[["raw", "cleaned"]].copy()
    return preview.to_dict(orient="records"), list(preview.columns)


# =========================
# STATE
# =========================
STATE = {
    "csv_path": DEFAULT_CSV_PATH,
    "total_rows": 0,
    "dataset_ok": False,
    "dataset_msg": "",

    "preview_n": DEFAULT_PREVIEW_N,
    "preview_choices": PREVIEW_CHOICES,
    "preview_rows": None,
    "preview_cols": None,

    "params": {
        "max_rows": 50000,
        "max_len": 80,
        "top_k": 10,
        "max_depth": 200,
        "smooth_window": 3,
        "repeats": 7,
        "warmup": 2,
        "n_list": "",
        "prefixes": ""
    },

    # PLOTS
    "png_build": None,
    "png_iter": None,
    "png_rec": None,
    "png_both": None,

    # TABLES
    "res_cols": None,
    "res_head": None,
    "agg_cols": None,
    "agg": None,

    # DOWNLOADS
    "res_csv_bytes": None,
    "agg_csv_bytes": None,
}


def refresh_dataset_state():
    ok, msg, total = get_dataset_info(STATE["csv_path"])
    STATE["dataset_ok"] = ok
    STATE["dataset_msg"] = msg
    STATE["total_rows"] = total

    if ok:
        try:
            n = int(STATE.get("preview_n", DEFAULT_PREVIEW_N))
            rows, cols = load_preview(STATE["csv_path"], n)
            STATE["preview_rows"] = rows
            STATE["preview_cols"] = cols
        except Exception:
            STATE["preview_rows"] = None
            STATE["preview_cols"] = None
    else:
        STATE["preview_rows"] = None
        STATE["preview_cols"] = None


refresh_dataset_state()


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def index():
    refresh_dataset_state()
    return render_template("index.html", state=STATE)


@app.route("/set_path", methods=["POST"])
def set_path():
    path = request.form.get("csv_path", "").strip()
    if path:
        STATE["csv_path"] = path

    refresh_dataset_state()
    if STATE["dataset_ok"]:
        flash(f"Dataset OK. Total baris: {STATE['total_rows']}")
    else:
        flash(STATE["dataset_msg"])

    # reset hasil eksperimen
    STATE["png_build"] = None
    STATE["png_iter"] = None
    STATE["png_rec"] = None
    STATE["png_both"] = None
    STATE["res_head"] = None
    STATE["agg"] = None
    STATE["res_csv_bytes"] = None
    STATE["agg_csv_bytes"] = None

    return redirect(url_for("index"))


@app.route("/set_preview", methods=["POST"])
def set_preview():
    try:
        n = int(request.form.get("preview_n", str(DEFAULT_PREVIEW_N)))
    except ValueError:
        n = DEFAULT_PREVIEW_N

    if n not in PREVIEW_CHOICES:
        n = DEFAULT_PREVIEW_N

    STATE["preview_n"] = n
    refresh_dataset_state()

    if STATE["dataset_ok"]:
        flash(f"Preview diubah menjadi {n} baris.")
    else:
        flash("Dataset belum valid, preview tidak dapat ditampilkan.")

    return redirect(url_for("index"))


@app.route("/run", methods=["POST"])
def run_experiment():
    refresh_dataset_state()
    if not STATE["dataset_ok"]:
        flash("Dataset belum valid. Periksa path dataset.")
        return redirect(url_for("index"))

    try:
        max_rows = int(request.form.get("max_rows", "50000"))
        max_len = int(request.form.get("max_len", "80"))
        top_k = int(request.form.get("top_k", "10"))
        max_depth = int(request.form.get("max_depth", "200"))
        smooth_window = int(request.form.get("smooth_window", "3"))

        repeats = int(request.form.get("repeats", str(STATE["params"].get("repeats", 7))))
        warmup = int(request.form.get("warmup", str(STATE["params"].get("warmup", 2))))
    except ValueError:
        flash("Input parameter angka tidak valid.")
        return redirect(url_for("index"))

    repeats = max(1, min(repeats, 25))
    warmup = max(0, min(warmup, 10))

    n_list_text = request.form.get("n_list", "")
    prefixes_text = request.form.get("prefixes", "")

    N_list = parse_int_list(n_list_text)
    prefixes = parse_prefixes(prefixes_text)

    if not N_list:
        flash("N list kosong / tidak valid. Contoh: 1000,5000,10000")
        return redirect(url_for("index"))
    if not prefixes:
        flash("Prefix kosong. Isi minimal 1 baris prefix.")
        return redirect(url_for("index"))

    STATE["params"] = {
        "max_rows": max_rows,
        "max_len": max_len,
        "top_k": top_k,
        "max_depth": max_depth,
        "smooth_window": max(1, smooth_window),
        "repeats": repeats,
        "warmup": warmup,
        "n_list": n_list_text,
        "prefixes": prefixes_text
    }

    # clamp max_rows
    max_rows = min(max_rows, STATE["total_rows"])

    # load data (lebih stabil kalau dtype string)
    df = pd.read_csv(
        STATE["csv_path"],
        usecols=[REQUIRED_COL],
        dtype={REQUIRED_COL: "string"},
    ).head(max_rows).copy()

    df["clean_name"] = df[REQUIRED_COL].astype(str).apply(preprocess_text)
    df = df[df["clean_name"].str.len() > 0].reset_index(drop=True)

    words_all = [w[:max_len] for w in df["clean_name"].tolist()]

    # filter N
    N_list = [N for N in N_list if N <= len(words_all)]
    if not N_list:
        flash("Semua N lebih besar dari data dipakai (MAX_ROWS). Naikkan MAX_ROWS atau kecilkan N.")
        return redirect(url_for("index"))

    rows = []
    for N in N_list:
        # ---- BUILD timing (median) ----
        def _build_call():
            t = build_trie(words_all[:N])
            del t

        build_med, build_sd = timed_call_ms(_build_call, repeats=repeats, warmup=warmup)

        # build trie real untuk query
        trie = build_trie(words_all[:N])

        for p in prefixes:
            # iterative timing
            def it_call():
                trie.suggest_iterative(p, limit=top_k)

            it_med, it_sd = timed_call_ms(it_call, repeats=repeats, warmup=warmup)

            # recursive timing
            rec_med = float("nan")
            rec_sd = float("nan")
            rec_err = 0

            def rec_call():
                trie.suggest_recursive(p, limit=top_k, max_depth=max_depth)

            try:
                rec_med, rec_sd = timed_call_ms(rec_call, repeats=repeats, warmup=warmup)
            except RecursionError:
                rec_err = 1

            # returned count (fairness)
            it_out = trie.suggest_iterative(p, limit=top_k)
            try:
                rec_out = trie.suggest_recursive(p, limit=top_k, max_depth=max_depth)
            except RecursionError:
                rec_out = []

            rows.append({
                "N": N,
                "prefix": p,

                "build_ms_median": build_med,
                "build_ms_sd": build_sd,

                "iterative_ms_median": it_med,
                "iterative_ms_sd": it_sd,

                "recursive_ms_median": rec_med,
                "recursive_ms_sd": rec_sd,
                "recursive_recursion_error": rec_err,

                "returned_iterative": len(it_out),
                "returned_recursive": len(rec_out),
            })

        del trie
        gc.collect()

    res = pd.DataFrame(rows)

    # agregasi per N
    agg = res.groupby("N", as_index=False).agg(
        build_ms=("build_ms_median", "mean"),

        iterative_ms=("iterative_ms_median", "mean"),
        recursive_ms=("recursive_ms_median", "mean"),

        returned_iterative=("returned_iterative", "mean"),
        returned_recursive=("returned_recursive", "mean"),

        rec_error_rate=("recursive_recursion_error", "mean"),
    ).sort_values("N").reset_index(drop=True)

    w = STATE["params"]["smooth_window"]
    agg["build_smooth"] = agg["build_ms"].rolling(window=w, min_periods=1).mean()
    agg["iterative_smooth"] = agg["iterative_ms"].rolling(window=w, min_periods=1).mean()
    agg["recursive_smooth"] = agg["recursive_ms"].rolling(window=w, min_periods=1).mean()

    # === PLOTS ===
    STATE["png_build"] = plot_line(
        agg["N"], agg["build_ms"],
        xlabel="N", ylabel="Waktu Build Trie (ms)",
        title=f"Build Trie: Mean(median-ms) vs N (trend window={w})",
        points=True,
        smooth_y=agg["build_smooth"],
        label_raw="Build (raw)",
        label_trend="Build (trend)"
    )

    STATE["png_iter"] = plot_line(
        agg["N"], agg["iterative_ms"],
        xlabel="N", ylabel="Waktu Iteratif (ms)",
        title=f"Iteratif: Mean(median-ms) vs N (trend window={w})",
        points=True,
        smooth_y=agg["iterative_smooth"],
        label_raw="Iteratif (raw)",
        label_trend="Iteratif (trend)"
    )

    STATE["png_rec"] = plot_line(
        agg["N"], agg["recursive_ms"],
        xlabel="N", ylabel="Waktu Rekursif (ms)",
        title=f"Rekursif: Mean(median-ms) vs N (trend window={w})",
        points=True,
        smooth_y=agg["recursive_smooth"],
        label_raw="Rekursif (raw)",
        label_trend="Rekursif (trend)"
    )

    STATE["png_both"] = plot_overlay(
        agg["N"],
        agg["iterative_smooth"],
        agg["recursive_smooth"],
        xlabel="N", ylabel="Waktu (ms)",
        title=f"Gabungan: Iteratif vs Rekursif (trend window={w})",
        label1="Iteratif (trend)",
        label2="Rekursif (trend)"
    )

    # === TABLES ===
    STATE["res_cols"] = list(res.columns)
    STATE["res_head"] = res.head(200).to_dict(orient="records")

    agg_disp = agg[[
        "N",
        "build_ms", "build_smooth",
        "iterative_ms", "iterative_smooth",
        "recursive_ms", "recursive_smooth",
        "returned_iterative", "returned_recursive",
        "rec_error_rate"
    ]]
    STATE["agg_cols"] = list(agg_disp.columns)
    STATE["agg"] = agg_disp.to_dict(orient="records")

    # downloads
    STATE["res_csv_bytes"] = res.to_csv(index=False).encode("utf-8")
    STATE["agg_csv_bytes"] = agg_disp.to_csv(index=False).encode("utf-8")

    flash(f"Eksperimen selesai. (median={repeats}x, warmup={warmup})")
    return redirect(url_for("index"))


@app.route("/download/res.csv", methods=["GET"])
def download_res():
    data = STATE.get("res_csv_bytes")
    if not data:
        flash("Belum ada hasil. Jalankan eksperimen dulu.")
        return redirect(url_for("index"))
    return send_file(io.BytesIO(data), mimetype="text/csv", as_attachment=True, download_name="hasil_iterasi_manual.csv")


@app.route("/download/agg.csv", methods=["GET"])
def download_agg():
    data = STATE.get("agg_csv_bytes")
    if not data:
        flash("Belum ada hasil. Jalankan eksperimen dulu.")
        return redirect(url_for("index"))
    return send_file(io.BytesIO(data), mimetype="text/csv", as_attachment=True, download_name="hasil_agregasi_mean_per_N.csv")


if __name__ == "__main__":
    # buka: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
