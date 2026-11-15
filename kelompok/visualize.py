import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "referensi" / "dataset"
OUTPUT_DIR = SCRIPT_DIR / "output"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simulation import (
    SensorNode,
    SimConfig,
    generate_and_preprocess_input,
    simulate_protocol,
)

PROTOCOLS = [
    "Proposed (Prim MST + Mobile Sink)",
    "Static Kruskal MST (Paper Code)",
    "LEACH (Clustering)",
    "PEGASIS (Chain Multi-Hop)",
    "TEEN (Threshold Routing)",
]

SHORT_LABELS = {
    "Proposed (Prim MST + Mobile Sink)": "Proposed\n(Prim+MS)",
    "Static Kruskal MST (Paper Code)": "Static\nKruskal",
    "LEACH (Clustering)": "LEACH",
    "PEGASIS (Chain Multi-Hop)": "PEGASIS",
    "TEEN (Threshold Routing)": "TEEN",
}

PROTOCOL_COLORS = {
    "Proposed (Prim MST + Mobile Sink)": "#1b9e77",
    "Static Kruskal MST (Paper Code)": "#386cb0",
    "LEACH (Clustering)": "#d95f02",
    "PEGASIS (Chain Multi-Hop)": "#7570b3",
    "TEEN (Threshold Routing)": "#e7298a",
}


def configure_plot_style() -> None:
    """Applies consistent academic styling across all generated figures."""
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def load_reference_ecp_datasets() -> Dict[str, Dict[str, List[float]]]:
    """Loads the five supplementary ECP CSV files from referensi/dataset/."""
    files = [
        ("Prim's Algorithm (Paper)", "prim's_algorithm_ecp_values.csv", "#1b9e77"),
        ("Data Aggregation", "data_aggregation_ecp_values.csv", "#386cb0"),
        ("Dynamic Routing", "dynamic_routing_ecp_values.csv", "#7570b3"),
        ("Clustering", "clustering_ecp_values.csv", "#d95f02"),
        ("Load Balancing", "load_balancing_ecp_values.csv", "#e7298a"),
    ]
    data: Dict[str, Dict[str, Any]] = {}
    for label, filename, color in files:
        path = DATASET_DIR / filename
        scenarios: List[str] = []
        values: List[float] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenarios.append(row["ECP Value"].strip())
                values.append(float(row["Value"]))
        data[label] = {"scenarios": scenarios, "values": values, "color": color}
    return data


def plot_fig1_reference_csv(out_path: Path) -> None:
    """Tahap 0: Visualizes the supplementary ECP datasets from the reference paper."""
    ref_data = load_reference_ecp_datasets()
    labels = list(ref_data.keys())
    scenarios = ref_data[labels[0]]["scenarios"]
    x = np.arange(len(scenarios))
    width = 0.15

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    offsets = np.linspace(-2 * width, 2 * width, len(labels))

    for offset, name in zip(offsets, labels):
        vals = ref_data[name]["values"]
        color = ref_data[name]["color"]
        bars = ax_bar.bar(x + offset, vals, width=width, label=name, color=color, edgecolor="white", linewidth=0.6)
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.00025,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
            )

        ax_line.plot(x, vals, marker="o", linewidth=2.0, markersize=6, label=name, color=color)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(scenarios)
    ax_bar.set_ylabel("ECP Value (dari CSV Paper)")
    ax_bar.set_ylim(0, 0.0168)
    ax_bar.set_title("A. Perbandingan Nilai ECP per Skenario (Bar)")
    ax_bar.legend(loc="upper left", fontsize=8, ncol=2)

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(scenarios)
    ax_line.set_ylabel("ECP Value (dari CSV Paper)")
    ax_line.set_ylim(0.001, 0.0148)
    ax_line.set_title("B. Tren Nilai ECP Lintas Skenario (cf. Figure 6 Paper)")
    ax_line.legend(loc="upper left", fontsize=8, ncol=2)

    fig.suptitle(
        "Tahap 0 — Baseline Dataset Referensi Paper (referensi/dataset/*_ecp_values.csv)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fig2_preprocessing(
    nodes: List[SensorNode],
    debug_info: Dict[str, Any],
    cfg: SimConfig,
    out_path: Path,
) -> None:
    """Tahap 1: Demonstrates sensor data preprocessing and initial spatial topology."""
    raw = debug_info["raw_stream"]
    imputed = debug_info["imputed"]
    cleaned = debug_info["cleaned"]
    norm = debug_info["norm_readings"]
    idx = np.arange(len(raw))
    nan_mask = np.isnan(raw)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=180)
    ax_raw, ax_clean, ax_norm, ax_topo = axes.flat

    # Panel A: Raw signal with NaN indicators and Z-score clipping bounds
    ax_raw.plot(idx, raw, color="#4d4d4d", linewidth=1.2, label="Raw stream (Node 00)")
    for nan_i in idx[nan_mask]:
        ax_raw.axvline(nan_i, color="#e41a1c", alpha=0.28, linewidth=1.2)
    ax_raw.scatter(
        idx[nan_mask],
        np.full(np.sum(nan_mask), debug_info["col_mean"]),
        color="#e41a1c",
        marker="x",
        s=42,
        zorder=5,
        label=f"Missing NaN ({int(np.sum(nan_mask))} titik)",
    )
    ax_raw.axhline(debug_info["upper_bound"], color="#ff7f00", linestyle="--", linewidth=1.1, label=r"Batas $\mu \pm 2.5\sigma$")
    ax_raw.axhline(debug_info["lower_bound"], color="#ff7f00", linestyle="--", linewidth=1.1)
    ax_raw.set_title("A. Data Mentah Sensor (Raw + Missing Value)")
    ax_raw.set_xlabel("Indeks Sampel Waktu")
    ax_raw.set_ylabel("Nilai Sensor Mentah")
    ax_raw.legend(loc="upper right", fontsize=8)

    # Panel B: Mean imputation + Z-score clipping
    ax_clean.plot(idx, imputed, color="#999999", linestyle=":", linewidth=1.1, label="Setelah Mean Imputation")
    ax_clean.plot(idx, cleaned, color="#386cb0", linewidth=1.4, label="Setelah Z-Score Clipping")
    ax_clean.scatter(
        idx[nan_mask],
        cleaned[nan_mask],
        color="#1b9e77",
        s=26,
        zorder=5,
        label=f"Nilai terimputasi (mean={debug_info['col_mean']:.2f})",
    )
    ax_clean.set_title("B. Mean Imputation & Z-Score Clipping (|z| <= 2.5)")
    ax_clean.set_xlabel("Indeks Sampel Waktu")
    ax_clean.set_ylabel("Nilai Terkoreksi")
    ax_clean.legend(loc="upper right", fontsize=8)

    # Panel C: Min-Max Normalization [0, 1]
    ax_norm.plot(idx, norm, color="#1b9e77", linewidth=1.4, label="Min-Max Normalized [0, 1]")
    ax_norm.axhline(0.35, color="#e7298a", linestyle="--", linewidth=1.1, label="Hard threshold TEEN (0.35)")
    ax_norm.set_ylim(-0.08, 1.12)
    ax_norm.set_title("C. Normalisasi Min-Max ke Rentang [0, 1]")
    ax_norm.set_xlabel("Indeks Sampel Waktu")
    ax_norm.set_ylabel("Nilai Ternormalisasi")
    ax_norm.legend(loc="upper right", fontsize=8)

    # Panel D: Initial 30-node topology on 200x200 m grid
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    energies = [n.energy for n in nodes]
    sc = ax_topo.scatter(xs, ys, c=energies, cmap="YlGn", vmin=0.44, vmax=0.56, s=75, edgecolors="#222222", linewidth=0.8)
    for n in nodes:
        ax_topo.text(n.x + 2.2, n.y + 2.2, f"{n.id:02d}", fontsize=7, color="#333333")
    ax_topo.scatter([cfg.grid_size / 2.0], [cfg.grid_size / 2.0], marker="*", s=220, color="#d95f02", edgecolors="black", zorder=6, label="Sink Awal (100, 100)")
    cbar = fig.colorbar(sc, ax=ax_topo, fraction=0.046, pad=0.03)
    cbar.set_label("Energi Awal $E_0$ (J)")
    ax_topo.set_xlim(-6, cfg.grid_size + 8)
    ax_topo.set_ylim(-8, cfg.grid_size + 12)
    ax_topo.set_title(f"D. Topologi Awal {cfg.num_nodes} Node ({int(cfg.grid_size)}x{int(cfg.grid_size)} m)")
    ax_topo.set_xlabel("Posisi X (m)")
    ax_topo.set_ylabel("Posisi Y (m)")
    ax_topo.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Tahap 1 — Input Simulasi, Pra-pemrosesan Data Sensor, dan Topologi Awal",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_routing_snapshot(
    ax: plt.Axes,
    nodes: List[SensorNode],
    snapshot: Dict[str, Any],
    cfg: SimConfig,
    title: str,
    show_trajectory: bool = False,
) -> Any:
    """Helper to render a routing tree snapshot onto a matplotlib Axes."""
    id_map = {n.id: n for n in nodes}
    sink_x, sink_y = snapshot["sink_pos"]
    parent_map = snapshot["parent_map"]
    cluster_heads = set(snapshot["cluster_heads"])
    energies = snapshot["node_energies"]
    alive = snapshot["node_alive"]

    # Draw routing edges first
    for child_id, parent_id in parent_map.items():
        child = id_map[child_id]
        px, py = (sink_x, sink_y) if parent_id == -1 else (id_map[parent_id].x, id_map[parent_id].y)
        is_sink_link = parent_id == -1
        ax.plot(
            [child.x, px],
            [child.y, py],
            color="#d95f02" if is_sink_link else "#5a7d9a",
            linewidth=1.3 if is_sink_link else 0.9,
            alpha=0.75 if is_sink_link else 0.65,
            zorder=1,
        )

    # Optional Mobile Sink trajectory
    if show_trajectory and len(snapshot["sink_trajectory"]) > 1:
        traj = np.array(snapshot["sink_trajectory"])
        ax.plot(traj[:, 0], traj[:, 1], color="#e7298a", linestyle="--", linewidth=1.0, alpha=0.75, zorder=3, label="Jejak Mobile Sink")
        ax.scatter([traj[0, 0]], [traj[0, 1]], marker="o", s=45, color="#e7298a", alpha=0.7, zorder=4)

    # Alive vs Dead nodes
    alive_nodes = [n for n in nodes if alive[n.id]]
    dead_nodes = [n for n in nodes if not alive[n.id]]

    sc = ax.scatter(
        [n.x for n in alive_nodes],
        [n.y for n in alive_nodes],
        c=[energies[n.id] for n in alive_nodes],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=0.55,
        s=68,
        edgecolors="#222222",
        linewidth=0.75,
        zorder=4,
    )
    if dead_nodes:
        ax.scatter(
            [n.x for n in dead_nodes],
            [n.y for n in dead_nodes],
            marker="X",
            s=72,
            color="#777777",
            edgecolors="black",
            linewidth=0.6,
            zorder=5,
            label=f"Node Mati ({len(dead_nodes)})",
        )

    # Highlight Cluster Heads
    ch_nodes = [n for n in alive_nodes if n.id in cluster_heads]
    if ch_nodes:
        ax.scatter(
            [n.x for n in ch_nodes],
            [n.y for n in ch_nodes],
            facecolors="none",
            edgecolors="#d95f02",
            linewidth=1.8,
            s=145,
            zorder=5,
            label="Cluster Head",
        )

    # Draw Sink
    ax.scatter([sink_x], [sink_y], marker="*", s=230, color="#e7298a", edgecolors="black", zorder=6, label="Sink")

    ax.set_xlim(-6, cfg.grid_size + 8)
    ax.set_ylim(-8, cfg.grid_size + 12)
    ax.set_xlabel("Posisi X (m)")
    ax.set_ylabel("Posisi Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7.5)
    return sc


def plot_fig3_graph_and_topology(
    nodes: List[SensorNode],
    sim_results: Dict[str, Dict[str, Any]],
    cfg: SimConfig,
    out_path: Path,
) -> None:
    """Tahap 2: Adjacency distance matrix + MST routing comparison at Round 600."""
    n_count = len(nodes)
    dist_mat = np.zeros((n_count, n_count))
    for i in range(n_count):
        for j in range(n_count):
            dist_mat[i, j] = nodes[i].distance_to(nodes[j].x, nodes[j].y)

    fig, (ax_mat, ax_kruskal, ax_prim) = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=180)

    # Panel A: Distance adjacency matrix
    im = ax_mat.imshow(dist_mat, cmap="viridis", aspect="auto")
    cbar_mat = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.03)
    cbar_mat.set_label("Jarak Euclidean (m)")
    ax_mat.set_title("A. Matriks Jarak Antar-Node (cf. Fig. 5 Paper)")
    ax_mat.set_xlabel("ID Node Tujuan")
    ax_mat.set_ylabel("ID Node Asal")

    # Panel B: Static Kruskal MST at Round 600
    kruskal_snap = sim_results["Static Kruskal MST (Paper Code)"]["snapshots"][600]
    _draw_routing_snapshot(
        ax_kruskal,
        nodes,
        kruskal_snap,
        cfg,
        title="B. Static Kruskal MST (Ronde 600, Sink Statis)",
        show_trajectory=False,
    )

    # Panel C: Proposed Prim MST + Mobile Sink at Round 600
    prim_snap = sim_results["Proposed (Prim MST + Mobile Sink)"]["snapshots"][600]
    sc = _draw_routing_snapshot(
        ax_prim,
        nodes,
        prim_snap,
        cfg,
        title="C. Proposed Prim MST + Mobile Sink (Ronde 600)",
        show_trajectory=True,
    )
    cbar_e = fig.colorbar(sc, ax=ax_prim, fraction=0.046, pad=0.03)
    cbar_e.set_label("Sisa Energi Node (J)")

    fig.suptitle(
        "Tahap 2 — Pemodelan Graf Matriks Ketetanggaan & Pohon Perutean MST (Snapshot Ronde 600)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fig4_round_dynamics(
    sim_results: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Tahap 3: Multi-round dynamics across rounds 1-1500."""
    fig, (ax_alive, ax_energy, ax_pkts) = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=180)

    for proto in PROTOCOLS:
        res = sim_results[proto]
        hist = res["history"]
        rounds = [h["round"] for h in hist]
        alive = [h["alive_nodes"] for h in hist]
        total_e = [h["total_energy"] for h in hist]
        cum_pkts = [h["cum_delivered_packets"] / 1000.0 for h in hist]
        color = PROTOCOL_COLORS[proto]
        short = SHORT_LABELS[proto].replace("\n", " ")

        ax_alive.plot(rounds, alive, label=f"{short} (FND={res['FND (Round)']})", color=color, linewidth=1.8)
        ax_alive.axvline(res["FND (Round)"], color=color, linestyle=":", linewidth=1.0, alpha=0.55)

        ax_energy.plot(rounds, total_e, label=short, color=color, linewidth=1.8)
        ax_pkts.plot(rounds, cum_pkts, label=short, color=color, linewidth=1.8)

    ax_alive.set_title("A. Node Aktif vs. Ronde (FND & LND)")
    ax_alive.set_xlabel("Ronde Simulasi")
    ax_alive.set_ylabel("Jumlah Node Hidup")
    ax_alive.set_ylim(-1, 32)
    ax_alive.legend(loc="lower left", fontsize=7.8)

    ax_energy.set_title("B. Total Sisa Energi Jaringan vs. Ronde")
    ax_energy.set_xlabel("Ronde Simulasi")
    ax_energy.set_ylabel("Total Sisa Energi (J)")
    ax_energy.legend(loc="upper right", fontsize=7.8)

    ax_pkts.set_title("C. Akumulasi Paket Terkirim vs. Ronde")
    ax_pkts.set_xlabel("Ronde Simulasi")
    ax_pkts.set_ylabel("Paket Terkirim (ribuan)")
    ax_pkts.legend(loc="upper left", fontsize=7.8)

    fig.suptitle(
        "Tahap 3 — Dinamika Simulasi Radio Multi-Ronde (Ronde 1–1500)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars: Any, fmt: str, y_offset_ratio: float = 0.02) -> None:
    """Places crisp numeric value labels above bars."""
    ylim_top = ax.get_ylim()[1]
    offset = ylim_top * y_offset_ratio
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_fig5_benchmark_evaluation(
    sim_results: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Tahap 4: Benchmark bar comparison across all 5 evaluation metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.8), dpi=180)
    ax_ecp, ax_ee, ax_life, ax_tput, ax_pdr, ax_tbl = axes.flat

    x = np.arange(len(PROTOCOLS))
    short_names = [SHORT_LABELS[p] for p in PROTOCOLS]
    colors = [PROTOCOL_COLORS[p] for p in PROTOCOLS]

    # 1. ECP (lower is better)
    ecp_vals = [sim_results[p]["ECP (mJ/pkt)"] for p in PROTOCOLS]
    bars = ax_ecp.bar(x, ecp_vals, color=colors, width=0.58, edgecolor="white")
    ax_ecp.set_ylim(0, max(ecp_vals) * 1.18)
    _annotate_bars(ax_ecp, bars, "{:.3f}")
    ax_ecp.set_xticks(x)
    ax_ecp.set_xticklabels(short_names, fontsize=8)
    ax_ecp.set_ylabel("mJ / paket")
    ax_ecp.set_title("A. ECP (Lebih Rendah Lebih Baik)")

    # 2. Energy Efficiency (higher is better)
    ee_vals = [sim_results[p]["EE (pkt/J)"] for p in PROTOCOLS]
    bars = ax_ee.bar(x, ee_vals, color=colors, width=0.58, edgecolor="white")
    ax_ee.set_ylim(0, max(ee_vals) * 1.18)
    _annotate_bars(ax_ee, bars, "{:.1f}")
    ax_ee.set_xticks(x)
    ax_ee.set_xticklabels(short_names, fontsize=8)
    ax_ee.set_ylabel("paket / Joule")
    ax_ee.set_title("B. Energy Efficiency (cf. Fig. 8 Paper)")

    # 3. Network Lifetime (FND & LND)
    fnd_vals = [sim_results[p]["FND (Round)"] for p in PROTOCOLS]
    lnd_vals = [sim_results[p]["LND (Round)"] for p in PROTOCOLS]
    w = 0.36
    bars_fnd = ax_life.bar(x - w / 2, fnd_vals, width=w, color="#1b9e77", edgecolor="white", label="FND")
    bars_lnd = ax_life.bar(x + w / 2, lnd_vals, width=w, color="#386cb0", edgecolor="white", label="LND")
    ax_life.set_ylim(0, max(lnd_vals) * 1.18)
    _annotate_bars(ax_life, bars_fnd, "{:.0f}", y_offset_ratio=0.015)
    _annotate_bars(ax_life, bars_lnd, "{:.0f}", y_offset_ratio=0.015)
    ax_life.set_xticks(x)
    ax_life.set_xticklabels(short_names, fontsize=8)
    ax_life.set_ylabel("Ronde")
    ax_life.set_title("C. Network Lifetime: FND & LND (cf. Fig. 10)")
    ax_life.legend(loc="upper right", fontsize=8)

    # 4. Throughput (Mb)
    tput_vals = [sim_results[p]["Throughput (Mb)"] for p in PROTOCOLS]
    bars = ax_tput.bar(x, tput_vals, color=colors, width=0.58, edgecolor="white")
    ax_tput.set_ylim(0, max(tput_vals) * 1.18)
    _annotate_bars(ax_tput, bars, "{:.2f}")
    ax_tput.set_xticks(x)
    ax_tput.set_xticklabels(short_names, fontsize=8)
    ax_tput.set_ylabel("Megabits (Mb)")
    ax_tput.set_title("D. Total Throughput (cf. Fig. 11 Paper)")

    # 5. PDR (%)
    pdr_vals = [sim_results[p]["PDR (%)"] for p in PROTOCOLS]
    bars = ax_pdr.bar(x, pdr_vals, color=colors, width=0.58, edgecolor="white")
    ax_pdr.set_ylim(90, 101.2)
    for bar in bars:
        h = bar.get_height()
        ax_pdr.text(bar.get_x() + bar.get_width() / 2.0, h + 0.2, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)
    ax_pdr.set_xticks(x)
    ax_pdr.set_xticklabels(short_names, fontsize=8)
    ax_pdr.set_ylabel("Persentase (%)")
    ax_pdr.set_title("E. Packet Delivery Ratio (cf. Fig. 9 Paper)")

    # 6. Summary Table Panel
    ax_tbl.axis("off")
    ax_tbl.set_title("F. Ringkasan Tabel Hasil Simulasi", pad=10)
    col_labels = ["Protokol", "ECP", "EE", "FND", "LND", "T-put", "PDR"]
    table_names = {
        "Proposed (Prim MST + Mobile Sink)": "Proposed",
        "Static Kruskal MST (Paper Code)": "Kruskal",
        "LEACH (Clustering)": "LEACH",
        "PEGASIS (Chain Multi-Hop)": "PEGASIS",
        "TEEN (Threshold Routing)": "TEEN",
    }
    cell_rows = []
    for p in PROTOCOLS:
        r = sim_results[p]
        cell_rows.append([
            table_names[p],
            f"{r['ECP (mJ/pkt)']:.3f}",
            f"{r['EE (pkt/J)']:.1f}",
            f"{r['FND (Round)']}",
            f"{r['LND (Round)']}",
            f"{r['Throughput (Mb)']:.2f}",
            f"{r['PDR (%)']:.2f}%",
        ])
    table = ax_tbl.table(
        cellText=cell_rows,
        colLabels=col_labels,
        colWidths=[0.20, 0.12, 0.13, 0.11, 0.11, 0.14, 0.15],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.02, 1.55)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#2b3e50")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx == 1:
            cell.set_facecolor("#e8f6f1")

    fig.suptitle(
        "Tahap 4 — Evaluasi Kinerja Akhir: Proposed Prim's MST + Mobile Sink vs. Protokol Pembanding",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def export_benchmark_csv(sim_results: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Exports benchmark summary metrics to CSV."""
    fieldnames = [
        "Protocol",
        "ECP (mJ/pkt)",
        "EE (pkt/J)",
        "FND (Round)",
        "LND (Round)",
        "Throughput (Mb)",
        "PDR (%)",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for proto in PROTOCOLS:
            writer.writerow({k: sim_results[proto][k] for k in fieldnames})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    cfg = SimConfig()
    nodes, debug_info = generate_and_preprocess_input(cfg, seed=42, return_debug=True)

    sim_results: Dict[str, Dict[str, Any]] = {}
    for proto in PROTOCOLS:
        sim_results[proto] = simulate_protocol(
            proto,
            nodes,
            cfg,
            record_history=True,
            snapshot_rounds=(1, 600),
        )

    plot_fig1_reference_csv(OUTPUT_DIR / "fig1_referensi_ecp_paper.png")
    plot_fig2_preprocessing(nodes, debug_info, cfg, OUTPUT_DIR / "fig2_tahap1_preprocessing.png")
    plot_fig3_graph_and_topology(nodes, sim_results, cfg, OUTPUT_DIR / "fig3_tahap2_graf_dan_topologi.png")
    plot_fig4_round_dynamics(sim_results, OUTPUT_DIR / "fig4_tahap3_dinamika_ronde.png")
    plot_fig5_benchmark_evaluation(sim_results, OUTPUT_DIR / "fig5_tahap4_evaluasi_metrik.png")
    export_benchmark_csv(sim_results, OUTPUT_DIR / "benchmark_results.csv")

    print("Berhasil mengekspor seluruh visualisasi dan CSV ke:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
