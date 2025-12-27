import os
import re
import io
import time
import gc
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # server-safe
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename


# =========================
# CONFIG
# =========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    out = sorted(set([x for x in out if x > 0]))
    return out


def parse_prefixes(multiline: str) -> List[str]:
    lines = [x.strip() for x in multiline.splitlines() if x.strip()]
    cleaned = [preprocess_text(x) for x in lines]
    return [x for x in cleaned if x]


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
        res = []
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
# PLOT HELPERS
# =========================
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_line(x, y, xlabel, ylabel, title, label=None, show_points=True, smooth_y=None, smooth_label=None) -> bytes:
    fig = plt.figure()
    if show_points:
        plt.plot(x, y, "o", alpha=0.55, label=(label + " (raw)") if label else "raw")
    if smooth_y is not None:
        plt.plot(x, smooth_y, "-", linewidth=2.2, label=smooth_label or (label + " (trend)" if label else "trend"))
    elif label:
        plt.plot(x, y, "-", linewidth=2.0, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if label or smooth_y is not None:
        plt.legend()
    return fig_to_png_bytes(fig)


def plot_overlay(x, y1, y2, xlabel, ylabel, title, label1, label2, smooth1=None, smooth2=None) -> bytes:
    fig = plt.figure()
    if smooth1 is not None:
        plt.plot(x, smooth1, "-", linewidth=2.2, label=f"{label1} (trend)")
    else:
        plt.plot(x, y1, "-", linewidth=2.0, label=label1)

    if smooth2 is not None:
        plt.plot(x, smooth2, "-", linewidth=2.2, label=f"{label2} (trend)")
    else:
        plt.plot(x, y2, "-", linewidth=2.0, label=label2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    return fig_to_png_bytes(fig)


# =========================
# TEMPLATE FILTER
# =========================
@app.template_filter("b64")
def b64_filter(data: bytes) -> str:
    if not data:
        return ""
    return base64.b64encode(data).decode("utf-8")


# =========================
# SIMPLE SERVER STATE
# =========================
STATE = {
    "csv_path": None,
    "csv_name": None,
    "total_rows": 0,

    # last run params
    "params": {
        "max_rows": 50000,
        "max_len": 80,
        "top_k": 10,
        "max_depth": 200,
        "smooth_window": 3,
        "n_list": "",
        "prefixes": ""
    },

    # last results
    "res_head": None,     # list of dicts (head)
    "res_cols": None,     # list of columns
    "agg": None,          # list of dicts
    "agg_cols": None,     # list of columns

    # bytes for downloads
    "res_csv_bytes": None,
    "agg_csv_bytes": None,

    # png bytes
    "png_iter": None,
    "png_rec": None,
    "png_both": None,
}


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        state=STATE
    )


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or f.filename.strip() == "":
        flash("Pilih file CSV dulu.")
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    try:
        # hitung total rows (cukup cepat untuk 1 juta baris, tapi tetap lumayan)
        df = pd.read_csv(save_path, usecols=["product_name"])
        total = len(df)
    except Exception as e:
        flash(f"Gagal membaca CSV / kolom product_name: {e}")
        return redirect(url_for("index"))

    STATE["csv_path"] = save_path
    STATE["csv_name"] = filename
    STATE["total_rows"] = total

    # reset hasil sebelumnya
    STATE["res_head"] = None
    STATE["agg"] = None
    STATE["png_iter"] = None
    STATE["png_rec"] = None
    STATE["png_both"] = None
    STATE["res_csv_bytes"] = None
    STATE["agg_csv_bytes"] = None

    flash(f"Upload berhasil: {filename} | total baris: {total}")
    return redirect(url_for("index"))


@app.route("/run", methods=["POST"])
def run_experiment():
    if STATE["csv_path"] is None:
        flash("Upload CSV dulu.")
        return redirect(url_for("index"))

    # ambil param
    try:
        max_rows = int(request.form.get("max_rows", "50000"))
        max_len = int(request.form.get("max_len", "80"))
        top_k = int(request.form.get("top_k", "10"))
        max_depth = int(request.form.get("max_depth", "200"))
        smooth_window = int(request.form.get("smooth_window", "3"))
    except ValueError:
        flash("Input parameter angka tidak valid.")
        return redirect(url_for("index"))

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

    # simpan param
    STATE["params"] = {
        "max_rows": max_rows,
        "max_len": max_len,
        "top_k": top_k,
        "max_depth": max_depth,
        "smooth_window": max(1, smooth_window),
        "n_list": n_list_text,
        "prefixes": prefixes_text
    }

    # load data (batasi)
    try:
        df = pd.read_csv(STATE["csv_path"], usecols=["product_name"]).head(max_rows)
    except Exception as e:
        flash(f"Gagal membaca CSV: {e}")
        return redirect(url_for("index"))

    df["clean_name"] = df["product_name"].astype(str).apply(preprocess_text)
    df = df[df["clean_name"].str.len() > 0].reset_index(drop=True)

    words_all = [w[:max_len] for w in df["clean_name"].tolist()]

    # filter N agar <= len(words_all)
    N_list = [N for N in N_list if N <= len(words_all)]
    if not N_list:
        flash("Semua N lebih besar dari data yang dipakai (MAX_ROWS). Naikkan MAX_ROWS atau kecilkan N.")
        return redirect(url_for("index"))

    # run
    rows = []
    for N in N_list:
        trie = build_trie(words_all[:N])

        for p in prefixes:
            # iterative
            t0 = time.perf_counter()
            _ = trie.suggest_iterative(p, limit=top_k)
            t_it = (time.perf_counter() - t0) * 1000

            # recursive
            try:
                t1 = time.perf_counter()
                _ = trie.suggest_recursive(p, limit=top_k, max_depth=max_depth)
                t_rec = (time.perf_counter() - t1) * 1000
            except RecursionError:
                t_rec = float("nan")

            rows.append({
                "N": N,
                "prefix": p,
                "iterative_ms": t_it,
                "recursive_ms": t_rec
            })

        del trie
        gc.collect()

    res = pd.DataFrame(rows)
    agg = res.groupby("N", as_index=False).agg(
        iterative_ms=("iterative_ms", "mean"),
        recursive_ms=("recursive_ms", "mean"),
    ).sort_values("N").reset_index(drop=True)

    # smoothing (moving average ringan)
    w = max(1, smooth_window)
    agg["iterative_smooth"] = agg["iterative_ms"].rolling(window=w, min_periods=1).mean()
    agg["recursive_smooth"] = agg["recursive_ms"].rolling(window=w, min_periods=1).mean()

    # plots (raw points + trend)
    png_iter = plot_line(
        agg["N"], agg["iterative_ms"],
        xlabel="N", ylabel="Waktu Iteratif (ms)",
        title=f"Iteratif: Mean ms vs N (trend window={w})",
        label="Iteratif",
        show_points=True,
        smooth_y=agg["iterative_smooth"],
        smooth_label="Iteratif (trend)"
    )

    png_rec = plot_line(
        agg["N"], agg["recursive_ms"],
        xlabel="N", ylabel="Waktu Rekursif (ms)",
        title=f"Rekursif: Mean ms vs N (trend window={w})",
        label="Rekursif",
        show_points=True,
        smooth_y=agg["recursive_smooth"],
        smooth_label="Rekursif (trend)"
    )

    png_both = plot_overlay(
        agg["N"], agg["iterative_ms"], agg["recursive_ms"],
        xlabel="N", ylabel="Waktu (ms)",
        title=f"Gabungan: Iteratif vs Rekursif (trend window={w})",
        label1="Iteratif", label2="Rekursif",
        smooth1=agg["iterative_smooth"],
        smooth2=agg["recursive_smooth"]
    )

    # store results (head for display, full for download)
    STATE["res_cols"] = list(res.columns)
    STATE["res_head"] = res.head(200).to_dict(orient="records")  # tampilkan 200 baris

    # untuk tabel agregasi tampilkan tanpa kolom smooth (biar rapi), tapi download bisa termasuk smooth kalau mau
    agg_display = agg[["N", "iterative_ms", "recursive_ms", "iterative_smooth", "recursive_smooth"]]
    STATE["agg_cols"] = list(agg_display.columns)
    STATE["agg"] = agg_display.to_dict(orient="records")

    STATE["res_csv_bytes"] = res.to_csv(index=False).encode("utf-8")
    STATE["agg_csv_bytes"] = agg_display.to_csv(index=False).encode("utf-8")

    STATE["png_iter"] = png_iter
    STATE["png_rec"] = png_rec
    STATE["png_both"] = png_both

    flash("Eksperimen selesai. Grafik & tabel diperbarui.")
    return redirect(url_for("index"))


@app.route("/download/res.csv", methods=["GET"])
def download_res():
    data = STATE.get("res_csv_bytes")
    if not data:
        flash("Belum ada hasil. Jalankan eksperimen dulu.")
        return redirect(url_for("index"))
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="hasil_iterasi_manual.csv"
    )


@app.route("/download/agg.csv", methods=["GET"])
def download_agg():
    data = STATE.get("agg_csv_bytes")
    if not data:
        flash("Belum ada hasil. Jalankan eksperimen dulu.")
        return redirect(url_for("index"))
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="hasil_agregasi_mean_per_N.csv"
    )


if __name__ == "__main__":
    # buka: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
