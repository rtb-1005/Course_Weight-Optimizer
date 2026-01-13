# main.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

from utils import load_desired_courses, load_global_state
from typing import List


# =========================
# 路径（只改这里）
# =========================
DESIRED_JSON_PATH = "desired_courses.json"
GLOBAL_JSON_PATH = "global_state.json"


# =========================
# 制度硬参数（只保留一处）
# =========================
BUDGET = 105.0
MIN_BID = 5.0


# =========================
# 不确定性与数值参数
# =========================
EPS = 1e-9
DELTA = 0.05

# “终局总入场次数”的情景：围绕 s_bar=M/P 扰动
S_MULTS = [0.8, 1, 1.6]           # 保守/中性/激进
DESIGN_INDEX = 1                    # 用中性档做投标向量

# 枚举进入多少门课（竞争性课程部分）
K_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 21]

BISect_ITERS = 80


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_overlap_stats(global_state) -> Tuple[float, float, float]:
    """
    M = sum bidders over ALL courses
    s_bar = M / P
    KMAX = BUDGET / MIN_BID
    """
    P = global_state.grade_size
    M = 0.0
    for st in global_state.courses.values():
        M += float(st.bidders)
    s_bar = M / float(P)
    KMAX = BUDGET / MIN_BID
    return M, s_bar, KMAX


def predict_final_bidders(global_state, s_target: float) -> Dict[str, float]:
    """
    预测终局 bidders：目标总入场次数 = P*s_target
    delta 按当前热度（bidders占比）分摊。
    """
    P = global_state.grade_size
    courses = global_state.courses

    M_cur = 0.0
    for st in courses.values():
        M_cur += float(st.bidders)

    target_total = float(P) * float(s_target)
    delta = max(0.0, target_total - M_cur)

    pred: Dict[str, float] = {}

    if M_cur > 0.0:
        for cid, st in courses.items():
            w = float(st.bidders) / M_cur
            pred[cid] = min(float(P), float(st.bidders) + delta * w)
    else:
        # 极端：当前全为0，均分新增入场
        n = max(1, len(courses))
        for cid, st in courses.items():
            pred[cid] = min(float(P), float(st.bidders) + delta / n)

    return pred


def compute_alpha(capacity: int, rho: float, mu: float) -> float:
    """
    alpha = mu * ln(max(rho, 1+DELTA))
    rho = bidders / capacity
    """
    rho_eff = max(rho, 1.0 + DELTA)
    return mu * math.log(rho_eff)


def waterfill_allocate(
    course_ids: List[str],
    utilities: Dict[str, float],
    alphas: Dict[str, float],
    budget: float,
    min_bid: float,
) -> Dict[str, float]:
    k = len(course_ids)
    if k == 0:
        return {}
    if budget < min_bid * k - 1e-12:
        raise ValueError("budget is insufficient for min_bid * K")

    remaining = budget - min_bid * k
    if remaining <= 1e-12:
        return {cid: min_bid for cid in course_ids}

    a = {cid: alphas[cid] + EPS for cid in course_ids}

    def total_extra(nu: float) -> float:
        s = 0.0
        for cid in course_ids:
            u = max(utilities[cid], EPS)
            aj = a[cid]
            val = aj * math.log(u / max(nu * aj, EPS)) - min_bid
            if val > 0.0:
                s += val
        return s

    hi = 0.0
    for cid in course_ids:
        u = max(utilities[cid], EPS)
        aj = a[cid]
        bound = u / (aj * math.exp(min_bid / aj))
        hi = max(hi, bound)
    hi *= 1.0001
    lo = hi * 1e-12

    for _ in range(BISect_ITERS):
        mid = 0.5 * (lo + hi)
        if total_extra(mid) > remaining:
            lo = mid
        else:
            hi = mid

    nu_star = hi

    bids: Dict[str, float] = {}
    for cid in course_ids:
        u = max(utilities[cid], EPS)
        aj = a[cid]
        extra = aj * math.log(u / max(nu_star * aj, EPS)) - min_bid
        if extra < 0.0:
            extra = 0.0
        bids[cid] = min_bid + extra

    total_bid = sum(bids.values())
    diff = budget - total_bid
    if abs(diff) > 1e-6:
        add = diff / k
        for cid in course_ids:
            bids[cid] = max(min_bid, bids[cid] + add)

    return bids


def proxy_prob(b: float, alpha: float) -> float:
    return 1.0 - math.exp(-b / (alpha + EPS))

def _wcswidth_fallback(s: str, ambiguous_as_wide: bool = False) -> int:
    """
    中文显示宽度优化：
    - 中日韩全角/宽字符：按 2 列
    - 组合字符：按 0 列
    - East Asian Width = 'A'（Ambiguous）默认按 1，可通过 ambiguous_as_wide=True 改为 2
    """
    import unicodedata

    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        ea = unicodedata.east_asian_width(ch)
        if ea in ("W", "F"):
            w += 2
        elif ea == "A":
            w += 2 if ambiguous_as_wide else 1
        else:
            w += 1
    return w


try:
    from wcwidth import wcswidth as _wcswidth  # pip install wcwidth
except Exception:
    _wcswidth = None


def _disp_width(s: str) -> int:
    if _wcswidth is not None:
        # wcwidth 对终端显示宽度的处理更完整
        return _wcswidth(s)
    return _wcswidth_fallback(s, ambiguous_as_wide=False)


def _pad(s: str, width: int, align: str) -> str:
    """按显示宽度补空格。align: 'L' or 'R'"""
    w = _disp_width(s)
    pad = width - w
    if pad <= 0:
        return s
    if align == "R":
        return (" " * pad) + s
    return s + (" " * pad)


def print_table(headers: List[str], rows: List[List[str]], aligns: List[str]) -> None:
    assert len(headers) == len(aligns)
    ncol = len(headers)

    widths = [_disp_width(h) for h in headers]
    for r in rows:
        for j in range(ncol):
            widths[j] = max(widths[j], _disp_width(r[j]))

    header_line = " | ".join(_pad(headers[j], widths[j], "L") for j in range(ncol))
    sep_line = "-+-".join("-" * widths[j] for j in range(ncol))
    print(header_line)
    print(sep_line)

    for r in rows:
        line = " | ".join(_pad(r[j], widths[j], aligns[j]) for j in range(ncol))
        print(line)
def main() -> int:
    if not Path(DESIRED_JSON_PATH).exists():
        print(f"[ERROR] desired json not found: {DESIRED_JSON_PATH}")
        return 2
    if not Path(GLOBAL_JSON_PATH).exists():
        print(f"[ERROR] global state json not found: {GLOBAL_JSON_PATH}")
        return 2

    desired = load_desired_courses(DESIRED_JSON_PATH)
    global_state = load_global_state(GLOBAL_JSON_PATH)

    P = global_state.grade_size
    M, s_bar_raw, KMAX = compute_overlap_stats(global_state)

    if s_bar_raw < 1.0 - 1e-9:
        print(f"[WARN] s_bar=M/P={s_bar_raw:.4f} < 1.0. "
              f"Likely global_state does NOT include all courses, or bidders definition is not '>=MIN_BID'.")
    s_bar = clamp(s_bar_raw, 1.0, KMAX)

    # 你想选的课程与偏好
    utilities: Dict[str, float] = {}
    for pref in desired.preferences:
        if pref.course_id in global_state.courses:
            utilities[pref.course_id] = float(pref.utility)
        else:
            print(f"[WARN] course_id not found in global_state: {pref.course_id}")

    desired_ids = [cid for cid, u in utilities.items() if u > 0.0]
    if not desired_ids:
        print("[ERROR] no valid desired courses after filtering.")
        return 1

    # 构造三档“终局入场”情景：s -> mu -> bidders_pred -> alpha
    scenarios_s: List[float] = []
    scenarios_mu: List[float] = []
    bidders_pred_list: List[Dict[str, float]] = []
    alphas_list: List[Dict[str, float]] = []

    for mult in S_MULTS:
        s = clamp(s_bar * mult, 1.0, KMAX)
        mu = BUDGET / max(s, EPS)

        pred_bidders = predict_final_bidders(global_state, s)

        alphas: Dict[str, float] = {}
        for cid in desired_ids:
            st = global_state.courses[cid]
            rho = pred_bidders[cid] / float(st.capacity)
            alphas[cid] = compute_alpha(st.capacity, rho, mu)

        scenarios_s.append(s)
        scenarios_mu.append(mu)
        bidders_pred_list.append(pred_bidders)
        alphas_list.append(alphas)

    # ---- 特殊处理：鲁棒不满课程（所有情景下 pred_bidders <= capacity）----
    robust_safe: List[str] = []
    competitive: List[str] = []
    for cid in desired_ids:
        st = global_state.courses[cid]
        worst_pred = max(pred[cid] for pred in bidders_pred_list)
        if worst_pred <= float(st.capacity) + 1e-12:
            robust_safe.append(cid)
        else:
            competitive.append(cid)

    # 鲁棒不满课：入场券即“确定收益”，但要考虑预算占用 => 枚举选多少门
    robust_safe.sort(key=lambda x: utilities[x], reverse=True)


    def eval_competitive(budget_rem: float) -> Tuple[float, List[str], Dict[str, float]]:
        """在竞争性课程上，用 DESIGN 情景求最优入场集合与投标，返回 (obj, subset, bids)."""
        if budget_rem < MIN_BID - 1e-12:
            return 0.0, [], {}
        if not competitive:
            return 0.0, [], {}

        # 只对竞争性课程计算 alphas（DESIGN 情景）
        alphas_design = {cid: alphas_list[DESIGN_INDEX][cid] for cid in competitive}
        utilities_comp = {cid: utilities[cid] for cid in competitive}

        # 先按“入场券收益”排序再枚举 K
        base = []
        for cid in competitive:
            a = alphas_design[cid]
            p0 = proxy_prob(MIN_BID, a)
            base.append((utilities_comp[cid] * p0, cid))
        base.sort(reverse=True)

        max_k_feasible = int(budget_rem // MIN_BID)
        max_k_feasible = min(max_k_feasible, len(competitive))

        best_obj = -1e100
        best_subset: List[str] = []
        best_bids: Dict[str, float] = {}

        for k in K_CANDIDATES:
            k2 = min(k, max_k_feasible)
            if k2 <= 0:
                continue
            subset = [cid for _, cid in base[:k2]]
            bids = waterfill_allocate(subset, utilities_comp, alphas_design, budget_rem, MIN_BID)
            probs = {cid: proxy_prob(bids[cid], alphas_design[cid]) for cid in subset}
            obj = sum(utilities_comp[cid] * probs.get(cid, 0.0) for cid in subset)
            if obj > best_obj:
                best_obj = obj
                best_subset = subset
                best_bids = bids

        if best_obj < 0:
            best_obj = 0.0
        return best_obj, best_subset, best_bids

    # 枚举选择 t 门鲁棒不满课（每门成本=5，收益=utility），其余预算给竞争性课程
    best_total = -1e100
    best_t = 0
    best_comp_subset: List[str] = []
    best_comp_bids: Dict[str, float] = {}

    max_safe_feasible = int(BUDGET // MIN_BID)
    max_safe_feasible = min(max_safe_feasible, len(robust_safe))

    safe_prefix_sum = [0.0]
    for cid in robust_safe:
        safe_prefix_sum.append(safe_prefix_sum[-1] + utilities[cid])

    for t in range(0, max_safe_feasible + 1):
        budget_rem = BUDGET - MIN_BID * t
        comp_obj, comp_subset, comp_bids = eval_competitive(budget_rem)
        total_obj = safe_prefix_sum[t] + comp_obj
        if total_obj > best_total:
            best_total = total_obj
            best_t = t
            best_comp_subset = comp_subset
            best_comp_bids = comp_bids

    safe_selected = robust_safe[:best_t]

    # 汇总最终 bids（未入场的为0）
    final_bids: Dict[str, float] = {cid: 0.0 for cid in desired_ids}
    for cid in safe_selected:
        final_bids[cid] = MIN_BID
    for cid, b in best_comp_bids.items():
        final_bids[cid] = b

    # 输出：对 safe_selected 概率=1；对竞争性用三情景区间
    print("========== Allocation Result ==========")
    print(f"Desired JSON: {DESIRED_JSON_PATH}")
    print(f"Global  JSON: {GLOBAL_JSON_PATH}")
    print(f"P (grade_size) = {P}")
    print(f"M = sum bidders over ALL courses = {M:.1f}")
    print(f"s_bar = M/P = {s_bar_raw:.4f} (clamped to {s_bar:.4f}, KMAX={KMAX:.2f})")
    print(f"Budget W = {BUDGET:.2f}, MinBid = {MIN_BID:.2f}")
    print("")
    tags = ["conservative", "neutral", "aggressive"]
    print("Scenarios (s, mu=W/s):")
    for i in range(3):
        print(f"  - {tags[i]:12s}: s={scenarios_s[i]:.4f}, mu={scenarios_mu[i]:.4f}")
    print("")
    print(f"Robust-underfull courses selected = {len(safe_selected)} (bid=MIN_BID, prob=1)")
    print(f"Competitive courses selected       = {len(best_comp_subset)}")
    print("")

    # 统一打印（按 bid 降序）
        # ---- 统一打印（按 bid 降序），概率拆列 + 增加 tag ----
    table_rows: List[List[str]] = []

    for cid in desired_ids:
        st = global_state.courses[cid]
        bid = final_bids[cid]
        u = utilities[cid]
        cap = st.capacity
        m_now = st.bidders
        pmin = min(pred[cid] for pred in bidders_pred_list)
        pmax = max(pred[cid] for pred in bidders_pred_list)

        if bid <= 0.0:
            tag = "OUT"
            p0 = p1 = p2 = 0.0
        elif cid in safe_selected:
            tag = "SAFE"
            p0 = p1 = p2 = 1.0
        else:
            tag = "COMP"
            p0 = proxy_prob(bid, alphas_list[0][cid])
            p1 = proxy_prob(bid, alphas_list[1][cid])
            p2 = proxy_prob(bid, alphas_list[2][cid])

        table_rows.append([
            cid,
            f"{bid:.2f}",
            f"{u:.2f}",
            str(cap),
            str(m_now),
            f"{pmin:.1f}..{pmax:.1f}",
            f"{p0:.3f}",
            f"{p1:.3f}",
            f"{p2:.3f}",
            tag,
        ])

    # 按 bid 降序，其次按 utility 降序
    table_rows.sort(key=lambda r: (float(r[1]), float(r[2])), reverse=True)

    print_table(
        headers=["course_id", "bid", "utility", "cap", "bidders_now", "pred[min..max]", "p(con)", "p(neu)", "p(agg)", "tag"],
        rows=table_rows,
        aligns=["L", "R", "R", "R", "R", "R", "R", "R", "R", "R"],
    )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
