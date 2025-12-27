import os
import re
import io
import time
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # penting untuk server (tanpa GUI)
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename


# =========================
# CONFIG
# =========================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "trie-autocomplete-aka-secret"  # untuk flash message


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
    cleaned = [x for x in cleaned if x]
    return cleaned


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
def plot_line(x, y, xlabel, ylabel, title, label=None) -> bytes:
    fig = plt.figure()
    if label:
        plt.plot(x, y, marker="o", label=label)
        plt.legend()
    else:
        plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_overlay(x, y1, y2, xlabel, ylabel, title, label1, label2) -> bytes:
    fig = plt.figure()
    plt.plot(x, y1, marker="o", label=label1)
    plt.plot(x, y2, marker="o", label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================
# GLOBAL STATE (simple)
# =========================
# Simpan dataset yang terakhir diupload (path)
STATE = {
    "csv_path": None,
    "total_rows": 0,
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        csv_loaded=STATE["csv_path"] is not None,
        total_rows=STATE["total_rows"]
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

    # hitung total rows (ringan: load sebagian untuk count? kita load penuh untuk kepastian)
    try:
        df = pd.read_csv(save_path, usecols=["product_name"])
        total = len(df)
    except Exception as e:
        flash(f"Gagal membaca CSV/kolom product_name: {e}")
        return redirect(url_for("index"))

    STATE["csv_path"] = save_path
    STATE["total_rows"] = total
    flash(f"Upload berhasil: {filename} | total baris: {total}")
    return redirect(url_for("index"))


@app.route("/run", methods=["POST"])
def run_experiment():
    if STATE["csv_path"] is None:
        flash("Upload CSV dulu.")
        return redirect(url_for("index"))

    # ambil parameter
    max_rows = int(request.form.get("max_rows", "50000"))
    max_len = int(request.form.get("max_len", "80"))
    top_k = int(request.form.get("top_k", "10"))
    max_depth = int(request.form.get("max_depth", "200"))
    n_list_text = request.form.get("n_list", "")
    prefixes_text = request.form.get("prefixes", "")

    N_list = parse_int_list(n_list_text)
    prefixes = parse_prefixes(prefixes_text)

    if not N_list:
        flash("N list kosong / tidak valid. Isi contoh: 10,50,100,500")
        return redirect(url_for("index"))
    if not prefixes:
        flash("Prefix kosong. Isi minimal 1 baris prefix.")
        return redirect(url_for("index"))

    # load data (batasi)
    df = pd.read_csv(STATE["csv_path"], usecols=["product_name"]).head(max_rows)
    df["clean_name"] = df["product_name"].astype(str).apply(preprocess_text)
    df = df[df["clean_name"].str.len() > 0].reset_index(drop=True)

    words_all = [w[:max_len] for w in df["clean_name"].tolist()]

    # filter N agar <= jumlah data dipakai
    N_list = [N for N in N_list if N <= len(words_all)]
    if not N_list:
        flash("Semua N lebih besar dari data yang dipakai. Naikkan MAX_ROWS atau kecilkan N.")
        return redirect(url_for("index"))

    # jalankan eksperimen
    rows = []
    for N in N_list:
        trie = build_trie(words_all[:N])

        for p in prefixes:
            t0 = time.perf_counter()
            _ = trie.suggest_iterative(p, limit=top_k)
            t_it = (time.perf_counter() - t0) * 1000

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
    )

    # bikin 3 plot (PNG bytes)
    png_iter = plot_line(
        agg["N"], agg["iterative_ms"],
        xlabel="N", ylabel="Waktu Iteratif (ms)",
        title="Iteratif: Mean ms vs N"
    )
    png_rec = plot_line(
        agg["N"], agg["recursive_ms"],
        xlabel="N", ylabel="Waktu Rekursif (ms)",
        title="Rekursif: Mean ms vs N"
    )
    png_both = plot_overlay(
        agg["N"], agg["iterative_ms"], agg["recursive_ms"],
        xlabel="N", ylabel="Waktu (ms)",
        title="Gabungan: Iteratif vs Rekursif (Mean ms vs N)",
        label1="Iteratif", label2="Rekursif"
    )

    # simpan hasil ke memory untuk download
    csv_bytes = res.to_csv(index=False).encode("utf-8")
    agg_bytes = agg.to_csv(index=False).encode("utf-8")

    # render halaman hasil
    return render_template(
        "index.html",
        csv_loaded=True,
        total_rows=STATE["total_rows"],
        result_table=res.head(200).to_dict(orient="records"),
        result_cols=list(res.columns),
        agg_table=agg.to_dict(orient="records"),
        agg_cols=list(agg.columns),
        png_iter=png_iter,
        png_rec=png_rec,
        png_both=png_both,
        csv_bytes=csv_bytes,
        agg_bytes=agg_bytes,
        params={
            "max_rows": max_rows,
            "max_len": max_len,
            "top_k": top_k,
            "max_depth": max_depth,
            "n_list": n_list_text,
            "prefixes": prefixes_text
        }
    )


@app.route("/download_res", methods=["POST"])
def download_res():
    data = request.form.get("csv_data", "").encode("utf-8")
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="hasil_iterasi_manual.csv"
    )


@app.route("/download_agg", methods=["POST"])
def download_agg():
    data = request.form.get("csv_data", "").encode("utf-8")
    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="hasil_agregasi_mean_per_N.csv"
    )


if __name__ == "__main__":
    # akses: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
