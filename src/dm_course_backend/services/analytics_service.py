# Recommendation analytics for the admin pages (Recommendations + Hybrid evaluation)
# and the shop's "For you" list. Everything is computed with pandas from the events:
# data/events.csv + the live events the frontend sends to POST /events (kept in memory).
#
# Offline evaluation = time split: we "train" on the events before split_time and
# check if the recommenders guess what people looked at after it.

import numpy as np
import pandas as pd

from dm_course_backend.models.data import load_events, load_products
from dm_course_backend.models.recommendation import POPULARITY_WEIGHTS, build_tfidf_matrix, popularity_from_events, prepare_product_text, recommend_similar
from dm_course_backend.services.platform_service import COLUMNS, live_events, rec_list

K = 5
SPLIT_QUANTILE = 0.8  # first 80% of the time = train, last 20% = test
CANDIDATES = 50  # every recommender returns a ranked list this long, we cut it to K after filtering
# Hand-picked merge weights for the hybrid (not learned)
HYBRID_WEIGHTS = {"popularity": 0.2, "content": 0.3, "co_view": 0.3, "same_category": 0.2}
METHODS = ["popularity", "same_category", "content", "co_view", "hybrid"]


def all_events() -> pd.DataFrame:
    events = load_events()
    if not live_events:
        return events
    live = pd.DataFrame(live_events)
    live["timestamp"] = pd.to_datetime(live.get("timestamp"), errors="coerce").fillna(pd.Timestamp.now())
    return pd.concat([events, live], ignore_index=True)


def interactions(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["event_type"].isin(POPULARITY_WEIGHTS) & events["product_id"].notna()].copy()
    ev["product_id"] = ev["product_id"].astype(int)
    return ev.sort_values("timestamp")


class Recommenders:
    """The 4 simple recommenders + hybrid, "trained" on one set of events."""

    def __init__(self, train: pd.DataFrame):
        self.products = load_products().copy()
        self.products["product_text"] = self.products["product_id"].map(prepare_product_text(self.products))
        self.matrix, _ = build_tfidf_matrix(self.products)
        self.train = train
        pop = popularity_from_events(train)
        self.pop = self.products.assign(score=self.products["product_id"].map(pop).fillna(0.0)) \
            .sort_values(["score", "product_id"], ascending=[False, True])
        self.category = dict(zip(self.products["product_id"], self.products["category_id"]))
        ev = interactions(train)[["user_session", "product_id"]].drop_duplicates()
        pairs = ev.merge(ev, on="user_session")
        pairs = pairs[pairs["product_id_x"] != pairs["product_id_y"]]
        self.co_counts = pairs.groupby(["product_id_x", "product_id_y"]).size()
        self.cache: dict = {}

    def popularity(self, seed: int) -> list[int]:
        return [p for p in self.pop["product_id"].head(CANDIDATES + 1) if p != seed][:CANDIDATES]

    # same as his recommend_same_category, but reuses the popularity computed once
    def same_category(self, seed: int) -> list[int]:
        cat = self.category.get(seed)
        same = self.pop[(self.pop["category_id"] == cat) & (self.pop["product_id"] != seed)]
        return same["product_id"].head(CANDIDATES).tolist()

    def content(self, seed: int) -> list[int]:
        return recommend_similar(self.products, seed, self.matrix, self.train, limit=CANDIDATES)["product_id"].tolist()

    def co_view(self, seed: int) -> list[int]:
        if seed not in self.co_counts.index.get_level_values(0):
            return []
        s = self.co_counts.loc[seed].reset_index(name="n").sort_values(["n", "product_id_y"], ascending=[False, True])
        return s["product_id_y"].head(CANDIDATES).tolist()

    # weighted reciprocal-rank merge of the 4 lists
    def hybrid(self, seed: int) -> list[int]:
        scores: dict[int, float] = {}
        for name, w in HYBRID_WEIGHTS.items():
            for rank, p in enumerate(self.recommend(name, seed), start=1):
                scores[p] = scores.get(p, 0.0) + w / rank
        return [p for p, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))][:CANDIDATES]

    def recommend(self, method: str, seed: int) -> list[int]:
        key = (method, seed)
        if key not in self.cache:
            self.cache[key] = getattr(self, method)(seed)
        return self.cache[key]

    def top_k(self, method: str, seed: int, exclude=(), k: int = K) -> list[int]:
        return [p for p in self.recommend(method, seed) if p not in exclude][:k]


def ranking_metrics(recs: list[int], targets: set[int], k: int = K) -> dict:
    hits = [1 if p in targets else 0 for p in recs[:k]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(targets), k)))
    return {"precision_at_k": sum(hits) / k, "recall_at_k": sum(hits) / len(targets),
            "hit_rate_at_k": float(any(hits)), "ndcg_at_k": dcg / idcg if idcg else 0.0}


def summarize(method: str, rows: list[dict], recommended: set[int], n_products: int) -> dict:
    df = pd.DataFrame(rows)
    out = {"method": method}
    for col in ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]:
        out[col] = round(float(df[col].mean()), 4) if not df.empty else None
    out["catalog_coverage"] = round(len(recommended) / n_products, 4)
    return out


def split(events: pd.DataFrame):
    # split time from the history file only, so live events always land in the test part
    split_time = load_events()["timestamp"].quantile(SPLIT_QUANTILE).floor("min")
    return split_time, events[events["timestamp"] < split_time], events[events["timestamp"] >= split_time]


_state: dict = {}


def trained():
    """Split + recommenders, rebuilt only when new live events arrive."""
    if _state.get("n_live") != len(live_events):
        events = all_events()
        split_time, train, test = split(events)
        _state.update(n_live=len(live_events), events=events, split_time=split_time, train=train, test=test,
                      recs=Recommenders(train), full=None)
    return _state


def session_cases(test: pd.DataFrame) -> tuple[list[tuple[int, set[int]]], int]:
    """Test sessions: seed = first product viewed, targets = the other products of that session."""
    ev = interactions(test)
    cases = []
    for _, g in ev.groupby("user_session", sort=True):
        products = list(dict.fromkeys(g["product_id"]))
        if len(products) >= 2:
            cases.append((products[0], set(products[1:])))
    return cases, int(test["user_session"].nunique())


# GET /analytics/recommendations
def offline_evaluation() -> dict:
    s = trained()
    recs: Recommenders = s["recs"]
    cases, n_sessions = session_cases(s["test"])
    comparison = []
    for method in METHODS:
        rows, recommended = [], set()
        for seed, targets in cases:
            top = recs.top_k(method, seed)
            recommended.update(top)
            rows.append(ranking_metrics(top, targets))
        comparison.append(summarize(method, rows, recommended, len(recs.products)))
    return {"split_time": str(s["split_time"]), "k": K, "n_test_cases": len(cases), "n_test_sessions": n_sessions,
            "n_train_events": len(s["train"]), "n_test_events": len(s["test"]), "comparison": comparison}


# GET /analytics/recommendations/hybrid
def hybrid_evaluation() -> dict:
    """Per user: seed = last product before the split, targets = new products after the split."""
    s = trained()
    recs: Recommenders = s["recs"]
    train, test = interactions(s["train"]), interactions(s["test"])
    train, test = train[train["user_id"].notna()], test[test["user_id"].notna()]
    seen = train.groupby("user_id")["product_id"].agg(lambda x: set(x))
    last = train.groupby("user_id")["product_id"].last()
    later = test.groupby("user_id")["product_id"].agg(lambda x: set(x))
    users = []
    for user_id, targets in later.items():
        if user_id in seen.index:
            new = targets - seen[user_id]
            if new:
                users.append((int(last[user_id]), seen[user_id], new))
    comparison = []
    for method in METHODS:
        rows, recommended = [], set()
        for seed, exclude, targets in users:
            top = recs.top_k(method, seed, exclude)
            recommended.update(top)
            rows.append(ranking_metrics(top, targets))
        comparison.append(summarize(method, rows, recommended, len(recs.products)))
    return {"split_time": str(s["split_time"]), "k": K, "n_test_users": len(users),
            "weights": HYBRID_WEIGHTS, "comparison": comparison}


def ctr_groups(df: pd.DataFrame, column: str, label=str) -> list[dict]:
    if df.empty:
        return []
    g = df.groupby(column)["event_type"].agg(
        impressions=lambda x: int((x == "recommendation_impression").sum()),
        clicks=lambda x: int((x == "recommendation_click").sum()))
    g["ctr"] = np.where(g["impressions"] > 0, (g["clicks"] / g["impressions"].where(g["impressions"] > 0)).round(4), 0.0)
    return [{"group": label(k), "impressions": int(r.impressions), "clicks": int(r.clicks), "ctr": float(r.ctr)}
            for k, r in g.iterrows()]


# GET /analytics/recommendations/ctr  (logged impressions and clicks)
def ctr_report() -> dict:
    ev = all_events()
    ev = ev[ev["event_type"].isin(["recommendation_impression", "recommendation_click"])].copy()
    impressions = int((ev["event_type"] == "recommendation_impression").sum())
    clicks = int((ev["event_type"] == "recommendation_click").sum())
    ev["recommendation_source"] = ev["recommendation_source"].fillna("unknown")
    pos = ev[ev["recommendation_position"].notna()].copy()
    pos["recommendation_position"] = pos["recommendation_position"].astype(int)
    return {"overall": {"group": "all", "impressions": impressions, "clicks": clicks,
                        "ctr": round(clicks / impressions, 4) if impressions else 0.0},
            "by_algorithm": {"groups": ctr_groups(ev, "recommendation_source")},
            "by_position": {"groups": ctr_groups(pos, "recommendation_position", lambda p: f"position {p}")}}


# GET /analytics/recommendations/session-comparison
def session_comparison() -> dict:
    """Simulation on the test sessions: after the first product, show K popular vs K personalized
    (hybrid) products; a "click" = the session really looked at that product later."""
    s = trained()
    recs: Recommenders = s["recs"]
    cases, _ = session_cases(s["test"])
    rows = []
    for name, method in [("popular", "popularity"), ("personalized (hybrid)", "hybrid")]:
        clicks = sum(len(set(recs.top_k(method, seed)) & targets) for seed, targets in cases)
        impressions = len(cases) * K
        rows.append({"algorithm": name, "impressions": impressions, "clicks": clicks,
                     "ctr": round(clicks / impressions, 4) if impressions else 0.0})
    return {"n_sessions": len(cases), "k": K, "simulation": True, "comparison": rows}


# GET /recommendations/user/{user_id}  (shop page "For you"): hybrid on ALL events, seeded by the user's last product
def user_recommendations(user_id: int, limit: int = 10) -> dict:
    s = trained()
    if s["full"] is None:
        s["full"] = Recommenders(s["events"])
    recs: Recommenders = s["full"]
    mine = interactions(s["events"])
    mine = mine[mine["user_id"] == user_id]
    if mine.empty:
        top = recs.pop.head(limit)
        return rec_list(top[COLUMNS].assign(score=top["score"].astype(float)), "popularity", True, "no_history")
    seen = set(mine["product_id"])
    ids = recs.top_k("hybrid", int(mine["product_id"].iloc[-1]), seen, k=limit)
    if len(ids) < limit:  # fill up with popular products
        ids += [p for p in recs.pop["product_id"] if p not in seen and p not in ids][:limit - len(ids)]
    df = recs.products.set_index("product_id").loc[ids].reset_index()
    df["score"] = [round(1 - i / len(ids), 4) for i in range(len(ids))]  # rank score, 1 = best
    return rec_list(df[COLUMNS], "hybrid")
