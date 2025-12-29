import cudf
import pandas as pd


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df


def create_recent_customer_candidates(
    transactions_df, recent_customer_weeks, customers=None
):
    if customers is not None:
        transactions_df = transactions_df[
            transactions_df["customer_id"].isin(customers)
        ]

    last_week_number = transactions_df["week_number"].max()

    recent_customer_df = (
        transactions_df.groupby(["customer_id", "article_id"])
        .agg(
            {
                "week_number": "max",
                "t_dat": "max",
                "price": "count",
            }
        )
        .rename(
            columns={
                "week_number": "ca_last_purchase_week",
                "t_dat": "ca_last_purchase_date",
                "price": "ca_purchase_count",
            }
        )
        .sort_values("ca_purchase_count", ascending=False)
    )

    features = (["customer_id", "article_id"], recent_customer_df)
    recent_customer_cand = (
        recent_customer_df.query(
            f"ca_last_purchase_week >= {last_week_number - recent_customer_weeks + 1}"
        )
        .reset_index()[["customer_id", "article_id"]]
        .drop_duplicates()
    )

    return recent_customer_cand, features


def create_last_customer_weeks_and_pairs(
    transactions_df, article_pairs_df, num_weeks, num_pair_weeks, customers
):
    clw_df = transactions_df[["customer_id", "article_id", "t_dat"]].copy()
    # thống kê giá để tính price-fit/discount cho cặp customer - article pair
    price_stats_df = (
        transactions_df.groupby("article_id")[["price"]]
        .agg(["mean", "max"])
        .reset_index()
    )
    price_stats_df.columns = [
        "article_id",
        "pair_article_mean_price",
        "pair_article_max_price",
    ]
    cust_price_stats_df = (
        transactions_df.groupby("customer_id")[["price"]]
        .agg(["mean", "max"])
        .reset_index()
    )
    cust_price_stats_df.columns = ["customer_id", "cust_price_mean", "cust_price_max"]
    if customers is not None:
        clw_df = clw_df[clw_df["customer_id"].isin(customers)]

    # only transactions in "x" weeks before last customer purchase
    last_customer_purchase_dat = clw_df.groupby("customer_id")["t_dat"].max()
    clw_df["max_cust_dat"] = clw_df["customer_id"].map(last_customer_purchase_dat)
    clw_df["sample"] = 1

    clw_df = (
        clw_df.groupby(["customer_id", "article_id"])
        .agg(
            {
                "max_cust_dat": "max",
                "sample": "count",
                "t_dat": "max",
            }
        )
        .rename(
            columns={
                "max_cust_dat": "last_c_purchase_date",
                "sample": "ca_count",
                "t_dat": "last_ca_purchase_date",
            }
        )
        .reset_index()
    )
    clw_df["last_ca_purchase_diff"] = (
        clw_df["last_c_purchase_date"] - clw_df["last_ca_purchase_date"]
    )

    clw_pairs_df = clw_df.query(
        f"last_ca_purchase_diff <= {num_pair_weeks * 7 - 1}"
    ).copy()
    clw_df = clw_df.query(f"last_ca_purchase_diff <= {num_weeks * 7 - 1}").copy()

    del last_customer_purchase_dat

    # merge with pairs, and get max of:
    #  - sources' last week(s) purchase count
    #  - count and percent of customer pairs (see generating code for details)
    clw_pairs_df = clw_pairs_df.merge(article_pairs_df, on="article_id")

    clw_pairs_df = (
        clw_pairs_df.groupby(["customer_id", "pair_article_id"])[
            [
                "ca_count",
                "last_ca_purchase_date",
                "last_ca_purchase_diff",
                "customer_count",
                "percent_customers",
                "pair_percent_customers",
                "lift",
            ]
        ]
        .max()
        .reset_index()
    )
    clw_pairs_df.columns = [
        "customer_id",
        "article_id",
        "pair_ca_count",
        "pair_last_ca_purchase_date",
        "pair_last_ca_purchase_diff",
        "pair_customer_count",
        "pair_percent_customers",
        "pair_reverse_percent_customers",
        "pair_lift",
    ]
    clw_pairs_df = clw_pairs_df.query("pair_customer_count > 2").copy()

    cust_last_week_cand = clw_df[["customer_id", "article_id"]].drop_duplicates()
    cust_last_week_pair_cand = clw_pairs_df[
        ["customer_id", "article_id"]
    ].drop_duplicates()

    pair_price_mean_map = price_stats_df.set_index("article_id")[
        "pair_article_mean_price"
    ]
    pair_price_max_map = price_stats_df.set_index("article_id")[
        "pair_article_max_price"
    ]
    cust_price_mean_map = cust_price_stats_df.set_index("customer_id")[
        "cust_price_mean"
    ]
    cust_price_max_map = cust_price_stats_df.set_index("customer_id")[
        "cust_price_max"
    ]

    clw_pairs_df["pair_article_mean_price"] = clw_pairs_df["article_id"].map(
        pair_price_mean_map
    )
    clw_pairs_df["pair_article_max_price"] = clw_pairs_df["article_id"].map(
        pair_price_max_map
    )
    clw_pairs_df["cust_price_mean"] = clw_pairs_df["customer_id"].map(
        cust_price_mean_map
    )
    clw_pairs_df["cust_price_max"] = clw_pairs_df["customer_id"].map(
        cust_price_max_map
    )
    clw_pairs_df["pair_price_fit"] = (
        clw_pairs_df["pair_article_mean_price"] - clw_pairs_df["cust_price_mean"]
    )
    clw_pairs_df["pair_discount_sensitivity"] = 1 - (
        clw_pairs_df["pair_article_mean_price"]
        / clw_pairs_df["pair_article_max_price"]
    )
    # fill giá trị mặc định cho các cặp ko có thống kê giá
    clw_pairs_df = clw_pairs_df.fillna(
        {
            "pair_price_fit": 0,
            "pair_discount_sensitivity": 0,
        }
    )

    clw_df = clw_df.set_index(["customer_id", "article_id"])[
        ["ca_count", "last_ca_purchase_date", "last_ca_purchase_diff"]
    ].copy()
    features = (["customer_id", "article_id"], clw_df)

    clw_pairs_df = clw_pairs_df.set_index(["customer_id", "article_id"])[
        [
            "pair_ca_count",
            "pair_last_ca_purchase_date",
            "pair_last_ca_purchase_diff",
            "pair_customer_count",
            "pair_percent_customers",
            "pair_reverse_percent_customers",
            "pair_lift",
            "pair_article_mean_price",
            "pair_article_max_price",
            "cust_price_mean",
            "cust_price_max",
            "pair_price_fit",
            "pair_discount_sensitivity",
        ]
    ].copy()
    pair_features = (["customer_id", "article_id"], clw_pairs_df)

    return cust_last_week_cand, cust_last_week_pair_cand, features, pair_features


def create_popular_article_cand(
    transactions_df,
    customers_df,
    articles_df,
    num_weeks,
    hier_col,
    num_candidates,
    num_articles=12,
    customers=None,
):
    ###########################################
    # first get general popular candidates
    ###########################################
    last_week_number = transactions_df["week_number"].max()

    # baseline
    article_purchases_df = (
        transactions_df.query(f"week_number >= {last_week_number - num_weeks + 1}")
        .groupby("article_id")["customer_id"]
        .count()
        .sort_values(ascending=False)
    )
    article_purchases_df = article_purchases_df.reset_index()
    article_purchases_df.columns = ["article_id", "counts"]
    popular_articles_df = article_purchases_df[:num_candidates].copy()
    popular_articles_df["join_col"] = 1

    # from here on, only care about relevant customers
    if customers is not None:
        transactions_df = transactions_df[
            transactions_df["customer_id"].isin(customers)
        ]
        customers_df = customers_df[customers_df["customer_id"].isin(customers)]

    popular_articles_cand = cudf.DataFrame(
        {"customer_id": customers_df["customer_id"], "join_col": 1}
    )
    popular_articles_cand = popular_articles_cand.merge(
        popular_articles_df, on="join_col"
    )
    del popular_articles_cand["join_col"]

    ###################################################
    # now let's limit it by cust/hierarchy information
    ###################################################
    sample_col = "t_dat"

    # add hierarchy column to transactions
    transactions_df[hier_col] = transactions_df["article_id"].map(
        articles_df.set_index("article_id")[hier_col]
    )
    # get customer/hierarchy statistics
    cust_hier = (
        transactions_df.groupby(["customer_id", hier_col])[sample_col]
        .count()
        .reset_index()
    )
    cust_hier.columns = list(cust_hier.columns)[:-1] + ["cust_hier_counts"]
    cust_hier = cust_hier.sort_values(
        ["customer_id", "cust_hier_counts"], ascending=False
    )
    cust_hier["total_counts"] = cust_hier["customer_id"].map(
        transactions_df.groupby("customer_id")[sample_col].count()
    )
    cust_hier["cust_hier_portion"] = (
        cust_hier["cust_hier_counts"] / cust_hier["total_counts"]
    )
    cust_hier = cust_hier[["customer_id", hier_col, "cust_hier_portion"]].copy()

    # add customer/hierarchy statistics to candidates
    popular_articles_cand[hier_col] = popular_articles_cand["article_id"].map(
        articles_df.set_index("article_id")[hier_col]
    )
    popular_articles_cand = popular_articles_cand.merge(
        cust_hier, on=["customer_id", hier_col], how="left"
    )
    popular_articles_cand["cust_hier_portion"] = popular_articles_cand[
        "cust_hier_portion"
    ].fillna(-1)

    del popular_articles_cand[hier_col]

    # take top based on customer/hierarchy statistics
    popular_articles_cand = popular_articles_cand.sort_values(
        ["customer_id", "cust_hier_portion", "counts"], ascending=False
    )
    popular_articles_cand = popular_articles_cand[["customer_id", "article_id"]].copy()
    popular_articles_cand = cudf_groupby_head(
        popular_articles_cand, "customer_id", num_articles
    )
    popular_articles_cand = popular_articles_cand.sort_values(
        ["customer_id", "article_id"]
    )
    popular_articles_cand = popular_articles_cand.reset_index(drop=True)

    # and save the article purchase statistics
    article_purchases_df = article_purchases_df[["article_id", "counts"]]
    article_purchases_df.columns = ["article_id", "recent_popularity_counts"]
    article_purchase_features = (
        ["article_id"],
        article_purchases_df.set_index("article_id"),
    )

    return popular_articles_cand, article_purchase_features


def create_age_bucket_candidates(
    transactions_df, customers_df, age_buckets, customers=None, articles=12
):
    # get transactions we're working with
    working_t_df = transactions_df.copy()
    working_t_df = working_t_df.drop_duplicates(
        ["customer_id", "article_id", "week_number"]
    )

    # create the buckets
    buckets_df = (
        customers_df[["customer_id"]].drop_duplicates().set_index("customer_id")
    )
    buckets_df["age"] = customers_df.set_index("customer_id").age
    buckets_df["age_bucket"] = pd.qcut(
        buckets_df["age"].to_pandas(), age_buckets
    ).cat.codes

    # choose bucket
    selected_buckets = ["age_bucket"]

    # add the buckets to the transactions
    working_t_df = working_t_df.merge(
        buckets_df[selected_buckets].reset_index(), on="customer_id"
    )

    # get the popularity
    last_week = working_t_df["week_number"].max()
    pi_df = (
        working_t_df.query(f"week_number=={last_week}")
        .groupby(selected_buckets + ["article_id"])["t_dat"]
        .count()
        .reset_index()
        .sort_values(selected_buckets + ["t_dat"], ascending=False)
    )
    pi_df = cudf_groupby_head(pi_df, selected_buckets, articles)

    # candidates - merge customer with their bucket
    can_df = buckets_df.reset_index()[["customer_id"] + selected_buckets].merge(
        pi_df, on=selected_buckets
    )
    can_df.columns = ["customer_id", "age_bucket", "article_id", "article_bucket_count"]

    # features dfs
    buckets_df = buckets_df[["age_bucket"]].copy()
    bucket_counts_df = can_df[
        ["customer_id", "article_id", "article_bucket_count"]
    ].copy()
    bucket_counts_df = bucket_counts_df.set_index(["customer_id", "article_id"])

    # candidates_df
    can_df = can_df[["customer_id", "article_id"]]
    if customers is not None:
        can_df = can_df[can_df["customer_id"].isin(customers)]

    return (
        can_df,
        (["customer_id"], buckets_df),
        (["customer_id", "article_id"], bucket_counts_df),
    )

def create_graph_based_embedding_candidates(
    transactions_df,
    num_weeks=12,
    label_week=None,                 # <-- thêm để CV không leakage
    embedding_dim=64,
    n_layers=2,
    epochs=3,
    lr=2e-3,
    weight_decay=1e-6,
    num_articles=200,
    min_transactions=3,
    customers=None,                  # customers to generate candidates for
    max_customers=None,
    sample_customers=False,
    use_cache=True,
    cache_path="lightgcn_cache.pt",
    batch_size=32768,                # BPR batch edges
    neg_k=1,
    retrieval_batch_users=256,       # retrieval batch users
    device=None,
    verbose=True,
):
    """
    Proper LightGCN (BPR-trained) for candidate retrieval.

    Returns:
      graph_candidates: DataFrame [customer_id, article_id]
      graph_features: (["customer_id","article_id"], feature_df indexed by keys)
                      columns: gcn_score, gcn_rank
    """
    import os
    import numpy as np

    # --- optional cudf support ---
    try:
        import cudf
        _HAS_CUDF = True
    except Exception:
        cudf = None
        _HAS_CUDF = False

    import pandas as pd
    import torch
    import torch.nn as nn

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # 1) Prepare training data (no leakage if label_week provided)
    # ----------------------------
    t = transactions_df
    if "cudf" in str(type(t)):
        t = t.to_pandas()
    else:
        t = t.copy()

    need_cols = ["customer_id", "article_id", "week_number"]
    missing = [c for c in need_cols if c not in t.columns]
    if missing:
        raise ValueError(f"transactions_df thiếu cột: {missing}")

    if label_week is None:
        # default: use last_week as "current" and train on last num_weeks weeks
        label_week = int(t["week_number"].max()) + 1

    start_week = int(label_week - num_weeks)
    end_week = int(label_week - 1)

    t = t[(t["week_number"] >= start_week) & (t["week_number"] <= end_week)]
    if customers is not None:
        # keep only these customers for training graph (optional)
        t = t[t["customer_id"].isin(customers)]

    # binary implicit edges
    t = t[["customer_id", "article_id"]].drop_duplicates()

    # filter customers by min_transactions (within window)
    cust_cnt = t.groupby("customer_id").size()
    valid_customers = cust_cnt[cust_cnt >= min_transactions].index

    if max_customers is not None and len(valid_customers) > max_customers:
        if sample_customers:
            rng = np.random.RandomState(42)
            valid_customers = rng.choice(valid_customers, size=max_customers, replace=False)
        else:
            valid_customers = cust_cnt.loc[valid_customers].nlargest(max_customers).index

    t = t[t["customer_id"].isin(valid_customers)]
    if len(t) == 0:
        empty = cudf.DataFrame(columns=["customer_id", "article_id"]) if _HAS_CUDF else pd.DataFrame(columns=["customer_id", "article_id"])
        empty_feat = cudf.DataFrame() if _HAS_CUDF else pd.DataFrame()
        return empty, (["customer_id", "article_id"], empty_feat)

    # factorize to contiguous indices
    u_codes, u_uniques = pd.factorize(t["customer_id"], sort=True)
    i_codes, i_uniques = pd.factorize(t["article_id"], sort=True)

    edge_u = u_codes.astype(np.int64)
    edge_i = i_codes.astype(np.int64)
    n_users = int(u_uniques.size)
    n_items = int(i_uniques.size)
    nnz = int(edge_u.shape[0])

    if verbose:
        print(f"[LightGCN] window weeks: [{start_week}, {end_week}] (label_week={label_week})")
        print(f"[LightGCN] edges(nnz)={nnz:,} | users={n_users:,} | items={n_items:,} | dim={embedding_dim} | layers={n_layers} | epochs={epochs}")
        print(f"[LightGCN] device={device}")

    # bought dict for filtering retrieval
    tmp_ui = pd.DataFrame({"u": edge_u, "i": edge_i})
    bought_dict = tmp_ui.groupby("u")["i"].apply(lambda x: np.unique(x.values)).to_dict()

    # ----------------------------
    # 2) Build normalized interaction (torch sparse COO): R_norm[u,i] = 1/sqrt(deg_u*deg_i)
    # ----------------------------
    def build_R_norm(edge_u_np, edge_i_np, U, I, dev):
        eu = torch.from_numpy(edge_u_np).to(dev)
        ei = torch.from_numpy(edge_i_np).to(dev)

        deg_u = torch.bincount(eu, minlength=U).float()
        deg_i = torch.bincount(ei, minlength=I).float()
        deg_u = torch.clamp(deg_u, min=1.0)
        deg_i = torch.clamp(deg_i, min=1.0)

        w = 1.0 / torch.sqrt(deg_u[eu] * deg_i[ei])
        idx = torch.stack([eu, ei], dim=0)
        return torch.sparse_coo_tensor(idx, w, size=(U, I), device=dev).coalesce()

    # ----------------------------
    # 3) LightGCN model (proper propagation) + BPR training
    # ----------------------------
    class _LightGCN(nn.Module):
        def __init__(self, U, I, D, K):
            super().__init__()
            self.U, self.I, self.D, self.K = U, I, D, K
            self.user_emb = nn.Embedding(U, D)
            self.item_emb = nn.Embedding(I, D)
            nn.init.normal_(self.user_emb.weight, std=0.02)
            nn.init.normal_(self.item_emb.weight, std=0.02)
            self.R_norm = None

        def set_R_norm(self, Rn):
            self.R_norm = Rn.coalesce()

        def propagate(self):
            u0 = self.user_emb.weight
            i0 = self.item_emb.weight
            u_list = [u0]
            i_list = [i0]
            u_k, i_k = u0, i0

            for _ in range(self.K):
                u_next = torch.sparse.mm(self.R_norm, i_k)
                i_next = torch.sparse.mm(self.R_norm.transpose(0, 1), u_k)
                u_k, i_k = u_next, i_next
                u_list.append(u_k)
                i_list.append(i_k)

            u_final = torch.stack(u_list, dim=0).mean(dim=0)
            i_final = torch.stack(i_list, dim=0).mean(dim=0)
            return u_final, i_final

    # cache key (avoid leakage / wrong reuse)
    cache_key = f"lw{label_week}_nw{num_weeks}_U{n_users}_I{n_items}_nnz{nnz}_d{embedding_dim}_L{n_layers}_ep{epochs}_lr{lr}"
    cache_obj = None
    if use_cache and os.path.exists(cache_path):
        try:
            ckpt = torch.load(cache_path, map_location="cpu")
            if isinstance(ckpt, dict) and ckpt.get("cache_key") == cache_key:
                cache_obj = ckpt
                if verbose:
                    print(f"[LightGCN] Loaded cache: {cache_path}")
        except Exception:
            cache_obj = None

    model = _LightGCN(n_users, n_items, embedding_dim, n_layers).to(device)
    R_norm = build_R_norm(edge_u, edge_i, n_users, n_items, device)
    model.set_R_norm(R_norm)

    if cache_obj is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        edge_u_t = torch.from_numpy(edge_u)
        edge_i_t = torch.from_numpy(edge_i)
        steps = (nnz + batch_size - 1) // batch_size

        model.train()
        for ep in range(1, epochs + 1):
            perm = torch.randperm(nnz)
            # propagate once per epoch (fast, common trick)
            u_final, i_final = model.propagate()

            total_loss = 0.0
            for st in range(steps):
                idx = perm[st * batch_size : (st + 1) * batch_size]
                u = edge_u_t[idx].to(device, non_blocking=True)
                i_pos = edge_i_t[idx].to(device, non_blocking=True)

                loss_acc = 0.0
                for _ in range(neg_k):
                    i_neg = torch.randint(0, n_items, (u.shape[0],), device=device)

                    uvec = u_final[u]
                    pvec = i_final[i_pos]
                    nvec = i_final[i_neg]

                    s_pos = (uvec * pvec).sum(dim=1)
                    s_neg = (uvec * nvec).sum(dim=1)

                    loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-8).mean()
                    loss_acc = loss_acc + loss

                opt.zero_grad(set_to_none=True)
                loss_acc.backward()
                opt.step()

                total_loss += float(loss_acc.detach().cpu())

                if verbose and (st + 1) % 200 == 0:
                    print(f"[LightGCN] ep {ep}/{epochs} step {st+1}/{steps} loss~ {total_loss/(st+1):.4f}")

            if verbose:
                print(f"[LightGCN] Epoch {ep} avg_loss={total_loss/steps:.4f}")

        # final embeddings
        model.eval()
        with torch.no_grad():
            u_final, i_final = model.propagate()

        if use_cache:
            try:
                torch.save(
                    {
                        "cache_key": cache_key,
                        "state_dict": model.state_dict(),
                        "u_final": u_final.detach().cpu(),
                        "i_final": i_final.detach().cpu(),
                        "u_uniques": u_uniques,
                        "i_uniques": i_uniques,
                        "bought_dict": bought_dict,
                    },
                    cache_path,
                )
                if verbose:
                    print(f"[LightGCN] Saved cache: {cache_path}")
            except Exception:
                if verbose:
                    print("[LightGCN] Warning: could not save cache.")
    else:
        # load cached embeddings + mappings
        model.load_state_dict(cache_obj["state_dict"])
        u_final = cache_obj["u_final"].to(device)
        i_final = cache_obj["i_final"].to(device)
        # use cached bought_dict / uniques (safer)
        u_uniques = cache_obj["u_uniques"]
        i_uniques = cache_obj["i_uniques"]
        bought_dict = cache_obj["bought_dict"]

    # ----------------------------
    # 4) Retrieval topK candidates
    # ----------------------------
    # users to generate for: if customers passed, restrict to those in mapping
    if customers is None:
        user_indices = np.arange(n_users, dtype=np.int64)
    else:
        # map given customer ids to u_idx (only those present in this window)
        # build reverse map once
        u_map = {cid: idx for idx, cid in enumerate(u_uniques)}
        user_indices = np.array([u_map[c] for c in customers if c in u_map], dtype=np.int64)

    item_T = i_final.t()  # (D, I)

    all_rows = []
    all_feats = []

    import pandas as pd

    for s in range(0, len(user_indices), retrieval_batch_users):
        ub = user_indices[s : s + retrieval_batch_users]
        ub_t = torch.from_numpy(ub).to(device)
        scores = (u_final[ub_t] @ item_T).float()  # (B, I)

        for bi, u_idx in enumerate(ub):
            sc = scores[bi]

            bought = bought_dict.get(int(u_idx), None)
            if bought is not None and len(bought) > 0:
                sc[bought] = -1e9

            k = min(num_articles, sc.numel())
            topv, topi = torch.topk(sc, k=k, largest=True, sorted=True)

            cust_id = u_uniques[u_idx]
            for rank, (it_idx, val) in enumerate(zip(topi.tolist(), topv.tolist()), start=1):
                art_id = i_uniques[it_idx]
                all_rows.append((cust_id, art_id))
                all_feats.append((cust_id, art_id, float(val), int(rank)))

    cand_pd = pd.DataFrame(all_rows, columns=["customer_id", "article_id"]).drop_duplicates()
    feat_pd = pd.DataFrame(
        all_feats, columns=["customer_id", "article_id", "gcn_score", "gcn_rank"]
    ).set_index(["customer_id", "article_id"])

    if _HAS_CUDF and "cudf" in str(type(transactions_df)):
        graph_candidates = cudf.from_pandas(cand_pd)
        graph_features_df = cudf.from_pandas(feat_pd.reset_index()).set_index(["customer_id", "article_id"])
        graph_features = (["customer_id", "article_id"], graph_features_df)
    else:
        graph_candidates = cand_pd
        graph_features = (["customer_id", "article_id"], feat_pd)

    if verbose:
        print(f"[LightGCN] Generated candidates: {len(graph_candidates):,}")

    return graph_candidates, graph_features


def add_features_to_candidates(candidates_df, features, customers_df, articles_df):
    """
    adds fields needed to merge in features
    and merges features in
    """
    for features_key in features:
        col_names, feature_df = features[features_key]

        # add the key to our df so we can merge the features in
        to_delete = []
        for col_name in col_names:
            if col_name not in candidates_df:
                if col_name in customers_df:
                    col_name_dict = customers_df.set_index("customer_id")[col_name]
                    candidates_df[col_name] = candidates_df["customer_id"].map(
                        col_name_dict
                    )
                    to_delete.append(col_name)
                elif col_name in articles_df:
                    col_name_dict = articles_df.set_index("article_id")[col_name]
                    candidates_df[col_name] = candidates_df["article_id"].map(
                        col_name_dict
                    )
                    to_delete.append(col_name)

        # now we can add the features
        candidates_df = candidates_df.merge(feature_df, how="left", on=col_names)

        for col_name in to_delete:
            del candidates_df[col_name]

    return candidates_df


def filter_candidates(candidates, transactions_df, **kwargs):
    recent_art_weeks = kwargs["filter_recent_art_weeks"]
    recent_articles = transactions_df.query(
        f"week_number >= {kwargs['label_week'] - recent_art_weeks}"
    )["article_id"]

    num_articles = kwargs.get("filter_num_articles", None)
    if num_articles is None:
        recent_articles = recent_articles.drop_duplicates()
    else:
        recent_item_counts = recent_articles.value_counts()
        most_popular_items = recent_item_counts[:num_articles].index
        most_popular_items = most_popular_items.to_pandas().to_list()
        recent_articles = most_popular_items

    candidates = candidates[candidates["article_id"].isin(recent_articles)].copy()

    return candidates
