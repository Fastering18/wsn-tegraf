import math
import heapq
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

# =====================================================================
# 1. SIMULATION PARAMETERS (From Table 1 in PMC11623227)
# =====================================================================
@dataclass
class SimConfig:
    num_nodes: int = 30               # Table 1: No. of nodes (static) = 30
    grid_size: float = 200.0          # Table 1: Grid topology size = 200 x 200 m
    max_rounds: int = 1500            # Simulation rounds
    packet_bits: int = 4000           # Bits per packet
    initial_energy: float = 0.5       # Initial battery per node (Joules)
    
    # First-Order Radio Dissipation Model
    e_elec: float = 50e-9             # 50 nJ/bit (Transceiver electronics)
    e_fs: float = 10e-12              # 10 pJ/bit/m^2 (Free-space amplifier)
    e_mp: float = 0.0013e-12          # 0.0013 pJ/bit/m^4 (Multi-path amplifier)
    e_da: float = 5e-9                # 5 nJ/bit/signal (Data aggregation)
    base_pdr_loss: float = 0.00015    # Distance-dependent channel loss factor

    @property
    def d0(self) -> float:
        """Threshold distance between free-space and multi-path models."""
        return math.sqrt(self.e_fs / self.e_mp)


@dataclass
class SensorNode:
    id: int
    x: float
    y: float
    energy: float
    readings: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    alive: bool = True

    def distance_to(self, other_x: float, other_y: float) -> float:
        return math.hypot(self.x - other_x, self.y - other_y)


# =====================================================================
# 2. DATA COLLECTION, IMPUTATION & NORMALIZATION PHASE
# =====================================================================
def generate_and_preprocess_input(
    cfg: SimConfig,
    seed: int = 42,
    return_debug: bool = False,
) -> Union[List[SensorNode], Tuple[List[SensorNode], Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    nodes = []
    debug_info: Dict[str, Any] = {}
    
    for i in range(cfg.num_nodes):
        x, y = rng.uniform(0, cfg.grid_size, size=2)
        # Simulate raw physical readings with 5% missing values (NaN) and outliers
        raw_stream = rng.normal(loc=25.0, scale=4.0, size=100)
        raw_stream[rng.random(100) < 0.05] = np.nan
        
        # Impute missing values using mean imputation
        col_mean = np.nanmean(raw_stream)
        imputed = np.where(np.isnan(raw_stream), col_mean, raw_stream)
        
        # Outlier clipping via Z-score (|z| > 2.5 clipped)
        mean_imp = float(np.mean(imputed))
        std_imp = float(np.std(imputed))
        lower_bound = mean_imp - 2.5 * std_imp
        upper_bound = mean_imp + 2.5 * std_imp
        z_scores = (imputed - mean_imp) / (std_imp + 1e-8)
        cleaned = np.clip(imputed, lower_bound, upper_bound)
        
        # Min-Max Normalization to [0, 1]
        norm_readings = (cleaned - np.min(cleaned)) / (np.ptp(cleaned) + 1e-8)
        
        # Heterogeneous slight variance in initial energy (0.45J - 0.55J)
        init_e = cfg.initial_energy * rng.uniform(0.9, 1.1)
        nodes.append(SensorNode(id=i, x=float(x), y=float(y), energy=float(init_e), readings=norm_readings))

        if return_debug and i == 0:
            debug_info = {
                "node_id": i,
                "raw_stream": raw_stream.copy(),
                "imputed": imputed.copy(),
                "cleaned": cleaned.copy(),
                "norm_readings": norm_readings.copy(),
                "col_mean": float(col_mean),
                "mean_imputed": mean_imp,
                "std_imputed": std_imp,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        
    if return_debug:
        return nodes, debug_info
    return nodes


# =====================================================================
# 3. RADIO ENERGY DISSIPATION MODEL
# =====================================================================
def calc_tx_energy(dist: float, bits: int, cfg: SimConfig) -> float:
    if dist < cfg.d0:
        return bits * cfg.e_elec + bits * cfg.e_fs * (dist ** 2)
    return bits * cfg.e_elec + bits * cfg.e_mp * (dist ** 4)

def calc_rx_energy(bits: int, cfg: SimConfig) -> float:
    return bits * cfg.e_elec


# =====================================================================
# 4. GRAPH ALGORITHMS: PROPOSED PRIM'S MST vs. AUTHOR'S KRUSKAL STUB
# =====================================================================
def build_energy_aware_prim_mst(
    nodes: List[SensorNode],
    sink_pos: Tuple[float, float],
    cfg: SimConfig
) -> Tuple[Dict[int, int], List[int]]:
    """
    Constructs an Energy-Balanced MST using Prim's Algorithm + Game-Theory Load Balancing.
    Returns parent_map {child_id: parent_id (-1 for sink)} and cluster_heads list.
    """
    alive_nodes = [n for n in nodes if n.alive]
    if not alive_nodes:
        return {}, []

    max_dist = math.hypot(cfg.grid_size, cfg.grid_size)
    
    # Phase A: Game-Theoretic Cluster Head (CH) Proposal & Selection
    # Players (nodes) evaluate payoff = Residual_Energy / Distance_to_Sink
    payoffs = {}
    for n in alive_nodes:
        d_sink = max(n.distance_to(*sink_pos), 1.0)
        payoffs[n.id] = (n.energy / cfg.initial_energy) ** 1.5 / (d_sink / max_dist)
        
    num_clusters = max(1, int(len(alive_nodes) * 0.15))
    sorted_by_payoff = sorted(alive_nodes, key=lambda n: payoffs[n.id], reverse=True)
    cluster_heads = [n.id for n in sorted_by_payoff[:num_clusters]]

    # Phase B: Prim's Algorithm from Virtual Sink Root (-1)
    # State tracking: (weight, u_parent, v_target)
    visited = {-1}
    parent_map: Dict[int, int] = {}
    degree_count: Dict[int, int] = {n.id: 0 for n in alive_nodes}
    degree_count[-1] = 0
    
    pq = []
    # Connect virtual sink (-1) directly to selected Cluster Heads first
    for n in alive_nodes:
        d_sink = n.distance_to(*sink_pos)
        e_penalty = 1.0 - (n.energy / (cfg.initial_energy * 1.1))
        # CHs get priority link to mobile sink
        ch_bonus = 0.35 if n.id in cluster_heads else 1.0
        w = ch_bonus * (0.55 * (d_sink / max_dist) + 0.45 * e_penalty)
        heapq.heappush(pq, (w, -1, n.id))

    id_to_node = {n.id: n for n in alive_nodes}

    while pq and len(visited) < len(alive_nodes) + 1:
        w, u, v = heapq.heappop(pq)
        if v in visited:
            continue
            
        visited.add(v)
        parent_map[v] = u
        degree_count[u] = degree_count.get(u, 0) + 1
        
        node_v = id_to_node[v]
        # Expand frontier from newly added node v to all unvisited alive nodes
        for nxt in alive_nodes:
            if nxt.id not in visited:
                dist = node_v.distance_to(nxt.x, nxt.y)
                norm_dist = dist / max_dist
                # Penalize routing through low-energy parent node_v or overloaded node_v
                energy_factor = 1.0 - (min(node_v.energy, nxt.energy) / (cfg.initial_energy * 1.1))
                load_factor = degree_count[v] / max(1, len(alive_nodes))
                
                edge_weight = 0.50 * norm_dist + 0.35 * energy_factor + 0.15 * load_factor
                heapq.heappush(pq, (edge_weight, v, nxt.id))

    return parent_map, cluster_heads


def optimize_mobile_sink_position(
    nodes: List[SensorNode],
    cluster_heads: List[int],
    cfg: SimConfig
) -> Tuple[float, float]:
    """Moves the mobile sink to the energy-weighted centroid of active Cluster Heads."""
    id_map = {n.id: n for n in nodes if n.alive}
    active_chs = [id_map[cid] for cid in cluster_heads if cid in id_map]
    if not active_chs:
        active_chs = [n for n in nodes if n.alive]
    if not active_chs:
        return (cfg.grid_size / 2.0, cfg.grid_size / 2.0)

    total_w = sum(n.energy for n in active_chs)
    if total_w <= 0:
        return (cfg.grid_size / 2.0, cfg.grid_size / 2.0)
        
    cx = sum(n.x * n.energy for n in active_chs) / total_w
    cy = sum(n.y * n.energy for n in active_chs) / total_w
    return (cx, cy)


def build_static_kruskal_mst(
    nodes: List[SensorNode],
    sink_pos: Tuple[float, float]
) -> Dict[int, int]:
    """
    Replicates the supplementary code's pure Euclidean Kruskal MST (Disjoint Set Union),
    then roots the tree at the stationary sink (-1).
    """
    alive_nodes = [n for n in nodes if n.alive]
    if not alive_nodes:
        return {}

    edges = []
    for n in alive_nodes:
        edges.append((n.distance_to(*sink_pos), -1, n.id))
    for i in range(len(alive_nodes)):
        for j in range(i + 1, len(alive_nodes)):
            u, v = alive_nodes[i], alive_nodes[j]
            edges.append((u.distance_to(v.x, v.y), u.id, v.id))
            
    edges.sort(key=lambda x: x[0])
    parent_dsu = {n.id: n.id for n in alive_nodes}
    parent_dsu[-1] = -1

    def find(i):
        if parent_dsu[i] == i:
            return i
        parent_dsu[i] = find(parent_dsu[i])
        return parent_dsu[i]

    adj = {n.id: [] for n in alive_nodes}
    adj[-1] = []
    for w, u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent_dsu[ru] = rv
            adj[u].append(v)
            adj[v].append(u)

    # BFS from Sink (-1) to orient edges toward sink
    parent_map = {}
    queue = [-1]
    visited = {-1}
    while queue:
        curr = queue.pop(0)
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                parent_map[nxt] = curr
                queue.append(nxt)
    return parent_map


# =====================================================================
# 5. MULTI-ROUND PROTOCOL SIMULATOR
# =====================================================================
def simulate_protocol(
    protocol_name: str,
    base_nodes: List[SensorNode],
    cfg: SimConfig,
    record_history: bool = False,
    snapshot_rounds: Tuple[int, ...] = (1, 600),
) -> Dict[str, Any]:
    nodes = [SensorNode(n.id, n.x, n.y, n.energy, n.readings.copy(), True) for n in base_nodes]
    id_map = {n.id: n for n in nodes}
    sink_pos = (cfg.grid_size / 2.0, cfg.grid_size / 2.0)  # Stationary default center
    
    fnd_round = cfg.max_rounds
    last_alive_round = 0
    total_packets_sent = 0
    total_packets_delivered = 0
    total_energy_consumed = 0.0
    rng = random.Random(100)
    history: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, Any]] = {}
    sink_trajectory: List[Tuple[float, float]] = [sink_pos]

    for r in range(1, cfg.max_rounds + 1):
        alive = [n for n in nodes if n.alive]
        if not alive:
            break
        last_alive_round = r

        # Build routing parent map {node_id: parent_id} based on protocol
        parent_map = {}
        round_chs: List[int] = []
        if protocol_name == "Proposed (Prim MST + Mobile Sink)":
            parent_map, chs = build_energy_aware_prim_mst(nodes, sink_pos, cfg)
            sink_pos = optimize_mobile_sink_position(nodes, chs, cfg)
            round_chs = list(chs)
            
        elif protocol_name == "Static Kruskal MST (Paper Code)":
            parent_map = build_static_kruskal_mst(nodes, sink_pos)
            
        elif protocol_name == "LEACH (Clustering)":
            p = 0.15
            chs = [n for n in alive if rng.random() < p] or [max(alive, key=lambda x: x.energy)]
            round_chs = [c.id for c in chs]
            for n in alive:
                if n in chs:
                    parent_map[n.id] = -1
                else:
                    nearest_ch = min(chs, key=lambda c: n.distance_to(c.x, c.y))
                    parent_map[n.id] = nearest_ch.id
                    
        elif protocol_name == "PEGASIS (Chain Multi-Hop)":
            # Greedy chain starting from farthest node
            unvisited = set(n.id for n in alive)
            curr = max(alive, key=lambda n: n.distance_to(*sink_pos)).id
            chain = [curr]
            unvisited.remove(curr)
            while unvisited:
                cn = id_map[curr]
                nxt = min(unvisited, key=lambda uid: cn.distance_to(id_map[uid].x, id_map[uid].y))
                chain.append(nxt)
                unvisited.remove(nxt)
                curr = nxt
            leader_idx = r % len(chain)
            leader_id = chain[leader_idx]
            round_chs = [leader_id]
            parent_map[leader_id] = -1
            for i in range(leader_idx):
                parent_map[chain[i]] = chain[i + 1]
            for i in range(len(chain) - 1, leader_idx, -1):
                parent_map[chain[i]] = chain[i - 1]

        elif protocol_name == "TEEN (Threshold Routing)":
            # Active sensing only when normalized reading delta crosses soft/hard threshold
            chs = sorted(alive, key=lambda x: x.energy, reverse=True)[:max(1, len(alive)//6)]
            round_chs = [c.id for c in chs]
            for n in alive:
                val = n.readings[r % len(n.readings)]
                if val < 0.35:  # Below hard threshold -> sleep / no tx
                    continue
                if n in chs:
                    parent_map[n.id] = -1
                else:
                    nearest_ch = min(chs, key=lambda c: n.distance_to(c.x, c.y))
                    parent_map[n.id] = nearest_ch.id

        # Execute Packet Transmission along Routing Tree
        for u_id, p_id in parent_map.items():
            u_node = id_map[u_id]
            if not u_node.alive:
                continue
                
            total_packets_sent += 1
            dist = u_node.distance_to(*sink_pos) if p_id == -1 else u_node.distance_to(id_map[p_id].x, id_map[p_id].y)
            
            e_tx = calc_tx_energy(dist, cfg.packet_bits, cfg)
            u_node.energy -= e_tx
            total_energy_consumed += e_tx

            if p_id != -1:
                p_node = id_map[p_id]
                if p_node.alive:
                    e_rx = calc_rx_energy(cfg.packet_bits, cfg) + cfg.packet_bits * cfg.e_da
                    p_node.energy -= e_rx
                    total_energy_consumed += e_rx

            # Channel success probability based on hop distance
            hop_loss_prob = min(0.25, cfg.base_pdr_loss * (dist ** 1.35))
            if rng.random() > hop_loss_prob:
                total_packets_delivered += 1

        # Check node deaths
        for n in nodes:
            if n.alive and n.energy <= 0:
                n.alive = False
                n.energy = 0.0
                if fnd_round == cfg.max_rounds:
                    fnd_round = r

        if record_history:
            sink_trajectory.append(sink_pos)
            energies_arr = [n.energy for n in nodes]
            history.append({
                "round": r,
                "alive_nodes": sum(1 for n in nodes if n.alive),
                "total_energy": float(sum(energies_arr)),
                "std_energy": float(np.std(energies_arr)),
                "cum_delivered_packets": total_packets_delivered,
                "sink_pos": sink_pos,
            })
            if r in snapshot_rounds:
                snapshots[r] = {
                    "round": r,
                    "parent_map": dict(parent_map),
                    "cluster_heads": list(round_chs),
                    "sink_pos": sink_pos,
                    "sink_trajectory": list(sink_trajectory),
                    "node_energies": {n.id: n.energy for n in nodes},
                    "node_alive": {n.id: n.alive for n in nodes},
                }

    # Compute Final Metrics (From Paper: ECP, EE, Lifetime, Throughput, PDR)
    pdr = (total_packets_delivered / max(1, total_packets_sent)) * 100.0
    # ECP: Energy expended per successfully transmitted packet (mJ / packet)
    ecp_mj = (total_energy_consumed * 1000.0) / max(1, total_packets_delivered)
    # EE: Packets delivered per Joule consumed
    ee_pkts_per_j = total_packets_delivered / max(1e-6, total_energy_consumed)
    throughput_mbits = (total_packets_delivered * cfg.packet_bits) / 1e6

    result: Dict[str, Any] = {
        "Protocol": protocol_name,
        "ECP (mJ/pkt)": round(ecp_mj, 3),
        "EE (pkt/J)": round(ee_pkts_per_j, 1),
        "FND (Round)": fnd_round,
        "LND (Round)": last_alive_round,
        "Throughput (Mb)": round(throughput_mbits, 2),
        "PDR (%)": round(pdr, 2),
    }
    if record_history:
        result["history"] = history
        result["snapshots"] = snapshots
    return result


if __name__ == "__main__":
    config = SimConfig()
    initial_nodes = generate_and_preprocess_input(config, seed=42)
    
    print("=== INPUT TOPOLOGY SUMMARY (Table 1 Parameters) ===")
    print(f"Nodes: {config.num_nodes} | Grid: {config.grid_size}x{config.grid_size}m | d0: {config.d0:.2f}m")
    for n in initial_nodes[:5]:
        print(f"  Node {n.id:02d}: pos=({n.x:6.2f}, {n.y:6.2f})m | E0={n.energy:.3f}J | NormReading[0]={n.readings[0]:.3f}")
    print("  ...\n")

    protocols = [
        "Proposed (Prim MST + Mobile Sink)",
        "Static Kruskal MST (Paper Code)",
        "LEACH (Clustering)",
        "PEGASIS (Chain Multi-Hop)",
        "TEEN (Threshold Routing)",
    ]

    print("=== BENCHMARK EVALUATION RESULTS ===")
    header = f"{'Protocol':<35} | {'ECP (mJ)':<9} | {'EE (pkt/J)':<10} | {'FND':<5} | {'LND':<5} | {'T-put (Mb)':<10} | {'PDR (%)'}"
    print(header)
    print("-" * len(header))
    for proto in protocols:
        res = simulate_protocol(proto, initial_nodes, config)
        print(f"{res['Protocol']:<35} | {res['ECP (mJ/pkt)']:<9.3f} | {res['EE (pkt/J)']:<10.1f} | "
              f"{res['FND (Round)']:<5} | {res['LND (Round)']:<5} | {res['Throughput (Mb)']:<10.2f} | {res['PDR (%)']:.2f}%")