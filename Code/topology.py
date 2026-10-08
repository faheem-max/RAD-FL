import numpy as np


class DynamicTopologyManager:
    """
    Dynamic Hardware-Aware D-Clique Topology Manager

    Implements the three proposed contributions:

    1. Dynamic Statistical Alignment
       - local model update vectors
       - cosine-based directional alignment
       - clique-average decentralized target proxy
       - EMA-smoothed target

    2. Physical Latency & Straggler Awareness
       - compute-time cost
       - communication latency
       - normalized statistical/hardware utility
       - P85 straggler filtering with connectivity fallback

    3. Bounded Topology Drift
       - topology changes are constrained by normalized
         Frobenius distance between consecutive mixing matrices

    IMPORTANT:
    This class does NOT use validation/test accuracy to choose topology.
    Topology learning uses only:
        - training-derived update vectors
        - hardware/network information
    """

    def __init__(
        self,
        num_clients=8,
        num_cliques=2,
        k_inter=2,
        mu=0.60,
        beta_g=0.20,
        straggler_quantile=0.85,
        delta_max=0.15,
        update_interval=5,
        sample_counts=None,
        aggregation_mode="metropolis",
        aggregation_alpha=0.5
    ):

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------
        if num_clients <= 0:
            raise ValueError(
                "num_clients must be positive."
            )

        if num_cliques <= 0:
            raise ValueError(
                "num_cliques must be positive."
            )

        if num_clients % num_cliques != 0:
            raise ValueError(
                "num_clients must be divisible "
                "by num_cliques."
            )

        if num_cliques > 1 and k_inter < 1:
            raise ValueError(
                "k_inter must be >= 1 when "
                "multiple cliques are used."
            )

        if not 0.0 <= mu <= 1.0:
            raise ValueError(
                "mu must be between 0 and 1."
            )

        if not 0.0 < beta_g <= 1.0:
            raise ValueError(
                "beta_g must be in (0, 1]."
            )

        if not 0.0 < straggler_quantile <= 1.0:
            raise ValueError(
                "straggler_quantile must be in (0, 1]."
            )

        if delta_max <= 0.0:
            raise ValueError(
                "delta_max must be positive."
            )

        if update_interval <= 0:
            raise ValueError(
                "update_interval must be positive."
            )

        # ----------------------------------------------------
        # Store configuration
        # ----------------------------------------------------
        self.num_clients = num_clients
        self.num_cliques = num_cliques

        self.clique_size = (
            num_clients // num_cliques
        )

        self.k_inter = k_inter

        self.mu = mu

        self.beta_g = beta_g

        self.straggler_quantile = (
            straggler_quantile
        )

        self.delta_max = delta_max

        self.update_interval = (
            update_interval
        )

        self.aggregation_mode = str(aggregation_mode).strip().lower()
        if self.aggregation_mode not in {"metropolis", "sample_aware", "hybrid"}:
            raise ValueError(
                "aggregation_mode must be 'metropolis', 'sample_aware', or 'hybrid'."
            )

        self.aggregation_alpha = float(aggregation_alpha)
        if self.aggregation_alpha < 0.0:
            raise ValueError("aggregation_alpha must be >= 0.")

        if sample_counts is None:
            if self.aggregation_mode in {"sample_aware", "hybrid"}:
                raise ValueError(
                    "sample_counts are required for sample_aware/hybrid aggregation."
                )
            self.sample_counts = None
        else:
            counts = np.asarray(sample_counts, dtype=np.float64).reshape(-1)
            if counts.size != self.num_clients:
                raise ValueError(
                    f"Expected {self.num_clients} sample counts, got {counts.size}."
                )
            if np.any(counts <= 0):
                raise ValueError("All sample counts must be positive.")
            self.sample_counts = counts

        self.eps = 1e-12


        self.cliques = []

        for clique_id in range(
            self.num_cliques
        ):

            start = (
                clique_id
                * self.clique_size
            )

            end = (
                start
                + self.clique_size
            )

            self.cliques.append(
                list(
                    range(
                        start,
                        end
                    )
                )
            )

        # ----------------------------------------------------
        # EMA target vector for each clique
        #
        # target_ema[clique_id] = vector
        # ----------------------------------------------------
        self.target_ema = {}

        # ----------------------------------------------------
        # Initial topology
        #
        # We use:
        #   - fully-connected intra-clique links
        #   - deterministic initial inter-clique shortcuts
        #
        # This avoids starting with disconnected cliques.
        # ----------------------------------------------------
        self.current_adj = (
            self._build_initial_adjacency()
        )

        self.current_W = (
            self._mixing_weights(
                self.current_adj
            )
        )

        # Last-update diagnostics for experiment logging
        self.last_metrics = {}

    # ========================================================
    # Clique utilities
    # ========================================================

    def _clique_of(self, client_id):
        return (
            client_id
            // self.clique_size
        )

    def _same_clique(
        self,
        i,
        j
    ):
        return (
            self._clique_of(i)
            ==
            self._clique_of(j)
        )

    # ========================================================
    # Initial D-Clique adjacency
    # ========================================================

    def _base_intra_clique_adjacency(self):
        """
        Fully connect every clique internally.

        Diagonal is kept zero here because self-weight is
        later generated by the Metropolis mixing matrix.
        """

        adj = np.zeros(
            (
                self.num_clients,
                self.num_clients
            ),
            dtype=np.float64
        )

        for members in self.cliques:

            for i in members:

                for j in members:

                    if i != j:
                        adj[i, j] = 1.0

        return adj

    def _build_initial_adjacency(self):
        """
        Build the initial connected topology using fully connected
        intra-group links and deterministic inter-group shortcuts.

        Each client receives deterministic initial shortcut
        edges so that the two cliques are connected from
        Round 1.

        These are only initialization shortcuts.
        Later topology updates replace them dynamically.
        """

        adj = (
            self._base_intra_clique_adjacency()
        )

        if self.num_cliques == 1:
            return adj

        # ----------------------------------------------------
        # Optimized initialization for our main 2-clique setup
        # ----------------------------------------------------
        if self.num_cliques == 2:

            left = self.cliques[0]
            right = self.cliques[1]

            max_offsets = min(
                self.k_inter,
                len(right)
            )

            for offset in range(
                max_offsets
            ):

                for position, i in enumerate(
                    left
                ):

                    j = right[
                        (
                            position
                            + offset
                        )
                        % len(right)
                    ]

                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        else:
            # ------------------------------------------------
            # General fallback for >2 cliques:
            # connect cliques in a chain using separate nodes.
            # ------------------------------------------------
            inter_degree = np.zeros(
                self.num_clients,
                dtype=int
            )

            for clique_id in range(
                self.num_cliques - 1
            ):

                left = self.cliques[
                    clique_id
                ]

                right = self.cliques[
                    clique_id + 1
                ]

                i = next(
                    x
                    for x in left
                    if inter_degree[x]
                    < self.k_inter
                )

                j = next(
                    x
                    for x in right
                    if inter_degree[x]
                    < self.k_inter
                )

                adj[i, j] = 1.0
                adj[j, i] = 1.0

                inter_degree[i] += 1
                inter_degree[j] += 1

        if not self._is_connected(adj):
            raise RuntimeError(
                "Initial topology is disconnected."
            )

        self._validate_inter_degree(
            adj
        )

        return adj

    # ========================================================
    # Metropolis-Hastings Mixing Matrix
    # ========================================================

    def _metropolis_hastings_weights(
        self,
        adj
    ):
        """
        Construct a symmetric doubly-stochastic mixing matrix.

        For physical edge (i,j):

            W_ij =
                1 / (1 + max(deg_i, deg_j))

        Self-weight:

            W_ii =
                1 - sum_{j != i} W_ij

        This preserves decentralized consensus mixing.
        """

        binary_adj = (
            adj > 0
        ).astype(
            np.float64
        )

        np.fill_diagonal(
            binary_adj,
            0.0
        )

        degrees = np.sum(
            binary_adj,
            axis=1
        )

        W = np.zeros_like(
            binary_adj,
            dtype=np.float64
        )

        for i in range(
            self.num_clients
        ):

            for j in range(
                i + 1,
                self.num_clients
            ):

                if binary_adj[i, j] > 0:

                    weight = (
                        1.0
                        /
                        (
                            1.0
                            + max(
                                degrees[i],
                                degrees[j]
                            )
                        )
                    )

                    W[i, j] = weight
                    W[j, i] = weight

        for i in range(
            self.num_clients
        ):

            W[i, i] = (
                1.0
                - np.sum(
                    W[i, :]
                )
            )

        self._validate_mixing_matrix(
            W
        )

        return W

    def _sample_aware_weights(
        self,
        adj
    ):
        """
        Construct a row-stochastic sample-aware mixing matrix.

        For client i, only its own model and currently connected neighbors
        participate. Their weights are proportional to local training sample
        counts:

            W_ij = n_j / (n_i + sum_{k in N_i} n_k)

        and

            W_ii = n_i / (n_i + sum_{k in N_i} n_k)

        This is intentionally row-stochastic, not necessarily symmetric or
        doubly stochastic. The physical topology remains undirected; only the
        aggregation influence is directional because neighborhood totals can
        differ across clients.
        """
        if self.sample_counts is None:
            raise RuntimeError(
                "sample_counts are not configured for sample-aware aggregation."
            )

        binary_adj = (adj > 0).astype(np.float64)
        np.fill_diagonal(binary_adj, 0.0)

        W = np.zeros_like(binary_adj, dtype=np.float64)

        for i in range(self.num_clients):
            active = [i] + [
                j for j in range(self.num_clients)
                if j != i and binary_adj[i, j] > 0
            ]
            denominator = float(np.sum(self.sample_counts[active]))
            if denominator <= 0.0:
                raise RuntimeError(
                    f"Invalid sample-aware denominator for client index {i}."
                )

            for j in active:
                W[i, j] = float(self.sample_counts[j] / denominator)

        self._validate_mixing_matrix(W)
        return W

    def _hybrid_weights(self, adj):
        """
        Construct a row-stochastic hybrid mixing matrix.

        Start from the Metropolis-Hastings matrix and reweight every active
        contributor (including self) by a sample-size factor:

            score_ij = W_ij^MH * (n_j / n_bar)^alpha

        followed by row normalization:

            W_ij = score_ij / sum_k score_ik

        alpha = 0 recovers the original Metropolis-Hastings matrix.
        Larger alpha gives more influence to clients with more local samples.
        The resulting matrix is row-stochastic but generally not symmetric.
        """
        if self.sample_counts is None:
            raise RuntimeError(
                "sample_counts are not configured for hybrid aggregation."
            )

        mh = self._metropolis_hastings_weights(adj)
        n_bar = float(np.mean(self.sample_counts))
        if n_bar <= 0.0:
            raise RuntimeError("Invalid mean sample count for hybrid aggregation.")

        sample_factor = np.power(
            self.sample_counts / n_bar,
            self.aggregation_alpha,
            dtype=np.float64,
        )

        scores = mh * sample_factor.reshape(1, -1)
        W = np.zeros_like(scores, dtype=np.float64)

        for i in range(self.num_clients):
            denom = float(np.sum(scores[i, :]))
            if denom <= 0.0:
                raise RuntimeError(
                    f"Invalid hybrid denominator for client index {i}."
                )
            W[i, :] = scores[i, :] / denom

        self._validate_mixing_matrix(W)
        return W

    def _mixing_weights(self, adj):
        """Dispatch to the configured aggregation-weight rule."""
        if self.aggregation_mode == "sample_aware":
            return self._sample_aware_weights(adj)
        if self.aggregation_mode == "hybrid":
            return self._hybrid_weights(adj)
        return self._metropolis_hastings_weights(adj)

    def _validate_mixing_matrix(
        self,
        W
    ):
        """
        Safety checks for the configured aggregation rule.

        All modes require non-negative row-stochastic weights.
        Metropolis-Hastings additionally requires symmetry and column
        stochasticity. Sample-aware and hybrid local weighting generally do
        not, because each row is normalized independently.
        """
        if not np.all(np.isfinite(W)):
            raise ValueError("W contains NaN or Inf values.")

        if np.min(W) < -1e-10:
            raise ValueError("W contains negative weights.")

        if not np.allclose(
            np.sum(W, axis=1),
            1.0,
            atol=1e-8
        ):
            raise ValueError("Rows of W do not sum to 1.")

        if self.aggregation_mode == "metropolis":
            if not np.allclose(W, W.T, atol=1e-8):
                raise ValueError("Metropolis mixing matrix W is not symmetric.")

            if not np.allclose(
                np.sum(W, axis=0),
                1.0,
                atol=1e-8
            ):
                raise ValueError("Columns of Metropolis W do not sum to 1.")

    # ========================================================
    # Graph connectivity utilities
    # ========================================================

    def _is_connected(
        self,
        adj
    ):
        """
        Standard DFS connectivity check.
        """

        visited = {0}
        stack = [0]

        while stack:

            node = stack.pop()

            neighbors = np.where(
                adj[node] > 0
            )[0]

            for neighbor in neighbors:

                neighbor = int(
                    neighbor
                )

                if (
                    neighbor != node
                    and neighbor not in visited
                ):

                    visited.add(
                        neighbor
                    )

                    stack.append(
                        neighbor
                    )

        return (
            len(visited)
            ==
            self.num_clients
        )

    def _components(
        self,
        adj
    ):
        """
        Return connected components.
        """

        remaining = set(
            range(
                self.num_clients
            )
        )

        components = []

        while remaining:

            start = next(
                iter(
                    remaining
                )
            )

            visited = {start}
            stack = [start]

            while stack:

                node = stack.pop()

                neighbors = np.where(
                    adj[node] > 0
                )[0]

                for neighbor in neighbors:

                    neighbor = int(
                        neighbor
                    )

                    if (
                        neighbor != node
                        and neighbor
                        not in visited
                    ):

                        visited.add(
                            neighbor
                        )

                        stack.append(
                            neighbor
                        )

            components.append(
                visited
            )

            remaining -= visited

        return components

    # ========================================================
    # Inter-clique degree / edge utilities
    # ========================================================

    def _inter_degree(
        self,
        adj,
        node
    ):

        node_clique = (
            self._clique_of(
                node
            )
        )

        degree = 0

        for j in range(
            self.num_clients
        ):

            if (
                adj[node, j] > 0
                and
                self._clique_of(j)
                != node_clique
            ):
                degree += 1

        return degree

    def _validate_inter_degree(
        self,
        adj
    ):

        for client_id in range(
            self.num_clients
        ):

            degree = (
                self._inter_degree(
                    adj,
                    client_id
                )
            )

            if degree > self.k_inter:

                raise ValueError(
                    f"Client {client_id} has "
                    f"{degree} inter-clique edges; "
                    f"k_inter={self.k_inter}."
                )

    def _inter_edges(
        self,
        adj
    ):
        """
        Return undirected inter-clique physical edges.
        """

        edges = set()

        for i in range(
            self.num_clients
        ):

            for j in range(
                i + 1,
                self.num_clients
            ):

                if (
                    adj[i, j] > 0
                    and
                    not self._same_clique(
                        i,
                        j
                    )
                ):

                    edges.add(
                        (i, j)
                    )

        return edges

    # ========================================================
    # Update Vector Processing
    # ========================================================

    def _to_numpy_updates(
        self,
        client_updates
    ):
        """
        Convert Torch / NumPy update vectors to consistent NumPy arrays.
        """

        updates = {}

        for client_id in range(
            self.num_clients
        ):

            if client_id not in client_updates:

                raise KeyError(
                    f"Missing update vector "
                    f"for client {client_id}."
                )

            vector = (
                client_updates[
                    client_id
                ]
            )

            if hasattr(
                vector,
                "detach"
            ):

                vector = (
                    vector
                    .detach()
                    .cpu()
                    .numpy()
                )

            vector = np.asarray(
                vector,
                dtype=np.float64
            ).reshape(-1)

            updates[
                client_id
            ] = vector

        dimensions = {
            vector.size
            for vector
            in updates.values()
        }

        if len(dimensions) != 1:

            raise ValueError(
                "All client update vectors must "
                "have identical dimensions."
            )

        return updates

    # ========================================================
    # Cosine Similarity
    # ========================================================

    def _cosine(
        self,
        a,
        b
    ):

        denominator = (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )

        if denominator <= self.eps:
            return 0.0

        return float(
            np.dot(a, b)
            /
            denominator
        )

    # ========================================================
    # NOVELTY 1
    # Dynamic Statistical Alignment
    # ========================================================

    def _update_target_ema(
        self,
        updates
    ):
        """
        Build decentralized target proxy for every clique.

        First compute:

            g_bar_C =
                mean(
                    Delta_theta_i
                    for i in clique C
                )

        Then update:

            g_target_C(t) =
                (1-beta_g) * g_target_C(t-1)
                +
                beta_g * g_bar_C(t)

        beta_g is configurable.

        IMPORTANT:
        This is a decentralized GLOBAL-TARGET PROXY,
        not an exact central-server global gradient.
        """

        for clique_id, members in enumerate(
            self.cliques
        ):

            clique_vectors = np.stack(
                [
                    updates[i]
                    for i in members
                ],
                axis=0
            )

            clique_mean = np.mean(
                clique_vectors,
                axis=0
            )

            if clique_id not in self.target_ema:

                self.target_ema[
                    clique_id
                ] = (
                    clique_mean.copy()
                )

            else:

                self.target_ema[
                    clique_id
                ] = (
                    (
                        1.0
                        - self.beta_g
                    )
                    *
                    self.target_ema[
                        clique_id
                    ]
                    +
                    self.beta_g
                    *
                    clique_mean
                )

    # ========================================================
    # NOVELTY 1 + NOVELTY 2
    # Candidate Edge Utility
    # ========================================================

    def _build_candidate_table(
        self,
        updates,
        hardware_profiles
    ):
        """
        Evaluate every possible INTER-CLIQUE edge.

        Statistical side:
        -----------------
        Pairwise cosine similarity is still calculated for
        diagnostics:

            cos(Delta_i, Delta_j)

        But the edge-selection score explicitly uses the
        proposed decentralized target proxy.

        For candidate pair (i,j):

            pair_update =
                0.5 * (Delta_i + Delta_j)

        We measure how well that pair direction aligns with
        each endpoint's clique target:

            A_ij =
                0.5 * [
                    cos(pair_update, target_Ci)
                    +
                    cos(pair_update, target_Cj)
                ]

        This operationalizes the proposed idea that a shortcut
        should make local communication direction more
        representative of the target/global learning direction.

        Hardware side:
        --------------
        Directional waiting cost:

            T(i <- j) =
                T_comp_j + latency_ij

        Because the physical edge is undirected, we use the
        worse of the two directional costs as its edge cost.
        """

        candidates = []

        raw_stat_scores = []
        raw_delay_scores = []

        for i in range(
            self.num_clients
        ):

            for j in range(
                i + 1,
                self.num_clients
            ):

                if self._same_clique(
                    i,
                    j
                ):
                    continue

                # --------------------------------------------
                # Original pairwise cosine metric
                # --------------------------------------------
                pairwise_cosine = (
                    self._cosine(
                        updates[i],
                        updates[j]
                    )
                )

                # --------------------------------------------
                # Target-relative pair alignment
                # --------------------------------------------
                pair_update = (
                    0.5
                    *
                    (
                        updates[i]
                        +
                        updates[j]
                    )
                )

                target_i = (
                    self.target_ema[
                        self._clique_of(i)
                    ]
                )

                target_j = (
                    self.target_ema[
                        self._clique_of(j)
                    ]
                )

                target_alignment_i = (
                    self._cosine(
                        pair_update,
                        target_i
                    )
                )

                target_alignment_j = (
                    self._cosine(
                        pair_update,
                        target_j
                    )
                )

                statistical_score = (
                    0.5
                    *
                    (
                        target_alignment_i
                        +
                        target_alignment_j
                    )
                )

                # --------------------------------------------
                # Hardware cost i <- j
                # --------------------------------------------
                compute_j = float(
                    hardware_profiles[
                        j
                    ].get(
                        "compute_time",
                        0.1
                    )
                )

                latency_ij = float(
                    hardware_profiles[
                        i
                    ].get(
                        "latency_matrix",
                        {}
                    ).get(
                        j,
                        0.05
                    )
                )

                directional_ij = (
                    compute_j
                    +
                    latency_ij
                )

                # --------------------------------------------
                # Hardware cost j <- i
                # --------------------------------------------
                compute_i = float(
                    hardware_profiles[
                        i
                    ].get(
                        "compute_time",
                        0.1
                    )
                )

                latency_ji = float(
                    hardware_profiles[
                        j
                    ].get(
                        "latency_matrix",
                        {}
                    ).get(
                        i,
                        0.05
                    )
                )

                directional_ji = (
                    compute_i
                    +
                    latency_ji
                )

                # Physical shortcut is undirected.
                edge_delay = max(
                    directional_ij,
                    directional_ji
                )

                candidate = {
                    "i": i,
                    "j": j,

                    "pairwise_cosine":
                        pairwise_cosine,

                    "statistical_score":
                        statistical_score,

                    "delay":
                        edge_delay
                }

                candidates.append(
                    candidate
                )

                raw_stat_scores.append(
                    statistical_score
                )

                raw_delay_scores.append(
                    edge_delay
                )

        if not candidates:

            return [], 0.0

        # ----------------------------------------------------
        # Min-max normalization over INTER-CLIQUE candidates.
        # ----------------------------------------------------
        stat_min = min(
            raw_stat_scores
        )

        stat_max = max(
            raw_stat_scores
        )

        delay_min = min(
            raw_delay_scores
        )

        delay_max = max(
            raw_delay_scores
        )

        # ----------------------------------------------------
        # P85 cutoff over INTER-CLIQUE candidates only.
        # ----------------------------------------------------
        cutoff = float(
            np.percentile(
                np.asarray(
                    raw_delay_scores,
                    dtype=np.float64
                ),
                self.straggler_quantile
                * 100.0
            )
        )

        for candidate in candidates:

            stat_norm = (
                (
                    candidate[
                        "statistical_score"
                    ]
                    -
                    stat_min
                )
                /
                (
                    stat_max
                    -
                    stat_min
                    +
                    self.eps
                )
            )

            delay_norm = (
                (
                    candidate[
                        "delay"
                    ]
                    -
                    delay_min
                )
                /
                (
                    delay_max
                    -
                    delay_min
                    +
                    self.eps
                )
            )

            # --------------------------------------------
            # Proposed joint statistical-hardware utility
            #
            # U(i,j) =
            #     mu * statistical_alignment
            #     -
            #     (1-mu) * hardware_delay
            # --------------------------------------------
            candidate[
                "utility"
            ] = (
                self.mu
                *
                stat_norm
                -
                (
                    1.0
                    - self.mu
                )
                *
                delay_norm
            )

            candidate[
                "allowed_by_p85"
            ] = (
                candidate[
                    "delay"
                ]
                <= cutoff
                + self.eps
            )

        # Highest utility first.
        #
        # pairwise cosine is only used as deterministic
        # secondary tie-breaker.
        candidates.sort(
            key=lambda x: (
                x["utility"],
                x["pairwise_cosine"]
            ),
            reverse=True
        )

        return (
            candidates,
            cutoff
        )

    # ========================================================
    # Proposed Physical Topology Construction
    # ========================================================

    def _construct_proposed_adjacency(
        self,
        candidates
    ):
        """
        Build candidate physical topology.

        Rules:

        1. All intra-clique edges remain.
        2. Full graph must remain connected.
        3. No client may exceed k_inter inter-clique edges.
        4. P85-filtered edges are normally excluded.
        5. If P85 would disconnect the graph, the best
           feasible filtered edge can be restored ONLY as a
           connectivity fallback.

        This preserves straggler awareness without allowing
        graph partitioning.
        """

        adj = (
            self._base_intra_clique_adjacency()
        )

        # ----------------------------------------------------
        # First establish global connectivity.
        # ----------------------------------------------------
        while not self._is_connected(
            adj
        ):

            components = (
                self._components(
                    adj
                )
            )

            component_id = {}

            for index, component in enumerate(
                components
            ):

                for node in component:

                    component_id[
                        node
                    ] = index

            feasible = []

            for candidate in candidates:

                i = candidate["i"]
                j = candidate["j"]

                if (
                    component_id[i]
                    ==
                    component_id[j]
                ):
                    continue

                if (
                    self._inter_degree(
                        adj,
                        i
                    )
                    >= self.k_inter
                ):
                    continue

                if (
                    self._inter_degree(
                        adj,
                        j
                    )
                    >= self.k_inter
                ):
                    continue

                feasible.append(
                    candidate
                )

            if not feasible:

                raise RuntimeError(
                    "Unable to construct a connected "
                    "topology under the current "
                    "k_inter degree constraint."
                )

            # Prefer P85-valid candidates.
            valid = [
                candidate
                for candidate
                in feasible
                if candidate[
                    "allowed_by_p85"
                ]
            ]

            if valid:
                selected = max(
                    valid,
                    key=lambda x:
                        x["utility"]
                )
            else:
                # Connectivity safety fallback:
                # choose best available edge even if
                # P85 filtered it.
                selected = max(
                    feasible,
                    key=lambda x:
                        x["utility"]
                )

            i = selected["i"]
            j = selected["j"]

            adj[i, j] = 1.0
            adj[j, i] = 1.0

        # ----------------------------------------------------
        # Fill remaining shortcut capacity using P85-valid
        # highest-utility edges.
        # ----------------------------------------------------
        for candidate in candidates:

            if not candidate[
                "allowed_by_p85"
            ]:
                continue

            i = candidate["i"]
            j = candidate["j"]

            if adj[i, j] > 0:
                continue

            if (
                self._inter_degree(
                    adj,
                    i
                )
                >= self.k_inter
            ):
                continue

            if (
                self._inter_degree(
                    adj,
                    j
                )
                >= self.k_inter
            ):
                continue

            adj[i, j] = 1.0
            adj[j, i] = 1.0

        # ----------------------------------------------------
        # Final safety checks
        # ----------------------------------------------------
        if not self._is_connected(
            adj
        ):

            raise RuntimeError(
                "Proposed topology is disconnected."
            )

        self._validate_inter_degree(
            adj
        )

        return adj

    # ========================================================
    # NOVELTY 3
    # Bounded Topology Drift
    # ========================================================

    def _frobenius_drift(
        self,
        W_new,
        W_reference=None
    ):
        """
        Normalized Frobenius topology drift:

            Delta_W =
                ||W_new - W_old||_F
                /
                sqrt(2N)
        """

        if W_reference is None:
            W_reference = self.current_W

        return float(
            np.linalg.norm(
                W_new
                -
                W_reference,
                ord="fro"
            )
            /
            np.sqrt(
                2.0
                *
                self.num_clients
            )
        )

    def _bounded_rewire(
        self,
        proposed_adj,
        utility_by_edge
    ):
        """
        Enforce the proposed Frobenius drift bound while also
        keeping PHYSICAL edges explicit.

        Why not directly use:
            W_new = (1-gamma)W_old + gamma W_proposed ?

        Since client communication uses
        active non-zero W_ij entries, those stale edges may
        remain physical neighbors.

        Therefore we preserve the SAME proposed contribution:

            normalized Frobenius drift <= delta_max

        but apply it to feasible physical rewiring.

        We gradually move the current adjacency toward the
        proposed adjacency and accept only changes whose
        resulting mixing matrix stays inside the
        Frobenius drift bound.

        This preserves:
          - explicit physical topology
          - strict k_inter degree cap
          - graph connectivity
          - valid row-stochastic W (symmetric/doubly-stochastic in Metropolis mode)
          - bounded topology drift
        """

        old_adj = (
            self.current_adj.copy()
        )

        old_W = (
            self.current_W.copy()
        )

        proposed_W = (
            self._mixing_weights(
                proposed_adj
            )
        )

        proposed_drift = (
            self._frobenius_drift(
                proposed_W,
                old_W
            )
        )

        # ----------------------------------------------------
        # If proposed topology already satisfies drift bound,
        # use it directly.
        # ----------------------------------------------------
        if (
            proposed_drift
            <= self.delta_max
            + self.eps
        ):

            return (
                proposed_adj,
                proposed_W,
                proposed_drift,
                proposed_drift
            )

        # ----------------------------------------------------
        # Otherwise gradually move toward the proposal.
        # ----------------------------------------------------
        working_adj = (
            old_adj.copy()
        )

        old_edges = (
            self._inter_edges(
                old_adj
            )
        )

        target_edges = (
            self._inter_edges(
                proposed_adj
            )
        )

        additions = list(
            target_edges
            -
            old_edges
        )

        # Prefer highest-utility desired edges first.
        additions.sort(
            key=lambda edge:
                utility_by_edge.get(
                    edge,
                    -np.inf
                ),
            reverse=True
        )

        # ----------------------------------------------------
        # Helper to validate a candidate transition.
        # ----------------------------------------------------
        def transition_is_valid(
            candidate_adj
        ):

            if not self._is_connected(
                candidate_adj
            ):
                return (
                    False,
                    None,
                    None
                )

            try:
                self._validate_inter_degree(
                    candidate_adj
                )
            except ValueError:
                return (
                    False,
                    None,
                    None
                )

            candidate_W = (
                self._mixing_weights(
                    candidate_adj
                )
            )

            drift = (
                self._frobenius_drift(
                    candidate_W,
                    old_W
                )
            )

            if (
                drift
                <= self.delta_max
                + self.eps
            ):

                return (
                    True,
                    candidate_W,
                    drift
                )

            return (
                False,
                candidate_W,
                drift
            )

        # ----------------------------------------------------
        # Try to add desired edges.
        #
        # If an endpoint already has k_inter shortcuts,
        # replace its lowest-utility obsolete edge.
        # ----------------------------------------------------
        for edge in additions:

            i, j = edge

            candidate_adj = (
                working_adj.copy()
            )

            edges_to_remove = set()

            for endpoint in (
                i,
                j
            ):

                if (
                    self._inter_degree(
                        candidate_adj,
                        endpoint
                    )
                    >= self.k_inter
                ):

                    removable = []

                    for existing_edge in (
                        self._inter_edges(
                            candidate_adj
                        )
                    ):

                        if (
                            endpoint
                            in existing_edge
                            and
                            existing_edge
                            not in target_edges
                        ):

                            removable.append(
                                existing_edge
                            )

                    if not removable:

                        edges_to_remove = None
                        break

                    # Remove the least useful obsolete edge.
                    removable.sort(
                        key=lambda existing_edge:
                            utility_by_edge.get(
                                existing_edge,
                                -np.inf
                            )
                    )

                    edges_to_remove.add(
                        removable[0]
                    )

            if edges_to_remove is None:
                continue

            for old_i, old_j in (
                edges_to_remove
            ):

                candidate_adj[
                    old_i,
                    old_j
                ] = 0.0

                candidate_adj[
                    old_j,
                    old_i
                ] = 0.0

            candidate_adj[
                i,
                j
            ] = 1.0

            candidate_adj[
                j,
                i
            ] = 1.0

            valid, _, _ = (
                transition_is_valid(
                    candidate_adj
                )
            )

            if valid:

                working_adj = (
                    candidate_adj
                )

        # ----------------------------------------------------
        # Remove obsolete edges if safe and still connected.
        # ----------------------------------------------------
        obsolete_edges = list(
            self._inter_edges(
                working_adj
            )
            -
            target_edges
        )

        for i, j in obsolete_edges:

            candidate_adj = (
                working_adj.copy()
            )

            candidate_adj[
                i,
                j
            ] = 0.0

            candidate_adj[
                j,
                i
            ] = 0.0

            valid, _, _ = (
                transition_is_valid(
                    candidate_adj
                )
            )

            if valid:

                working_adj = (
                    candidate_adj
                )

        final_W = (
            self._mixing_weights(
                working_adj
            )
        )

        actual_drift = (
            self._frobenius_drift(
                final_W,
                old_W
            )
        )

        return (
            working_adj,
            final_W,
            proposed_drift,
            actual_drift
        )

    # ========================================================
    # MAIN TOPOLOGY UPDATE
    # ========================================================

    def update_topology(
        self,
        round_num,
        client_updates,
        hardware_profiles
    ):
        """
        Called after client update vectors from the previous
        communication round are available.

        Important distinction:

        EMA target is updated EVERY round.

        Physical topology is optimized only when:

            round_num % update_interval == 0

        Example K=5:
            EMA updated each round
            topology changes at rounds 5,10,15,...
        """

        # ----------------------------------------------------
        # Convert training-derived local updates.
        # ----------------------------------------------------
        updates = (
            self._to_numpy_updates(
                client_updates
            )
        )

        # ----------------------------------------------------
        # Novelty 1:
        # update clique/global target proxy every round.
        # ----------------------------------------------------
        self._update_target_ema(
            updates
        )

        # ----------------------------------------------------
        # No physical topology change this round.
        # ----------------------------------------------------
        if (
            round_num
            % self.update_interval
            != 0
        ):

            self.last_metrics = {
                "round":
                    round_num,

                "topology_updated":
                    False,

                "actual_drift":
                    0.0,

                "delta_max":
                    self.delta_max,

                "connected":
                    self._is_connected(
                        self.current_adj
                    ),

                "inter_edge_count":
                    len(
                        self._inter_edges(
                            self.current_adj
                        )
                    ),

                "max_inter_degree":
                    max(
                        self._inter_degree(
                            self.current_adj,
                            i
                        )
                        for i
                        in range(
                            self.num_clients
                        )
                    ),

                "ema_target_norms":
                    {
                        str(clique_id):
                            float(
                                np.linalg.norm(
                                    target
                                )
                            )

                        for clique_id, target
                        in self.target_ema.items()
                    }
            }

            return (
                self.current_W
            )

        # ====================================================
        # Actual topology optimization
        # ====================================================

        candidates, p85_cutoff = (
            self._build_candidate_table(
                updates,
                hardware_profiles
            )
        )

        proposed_adj = (
            self._construct_proposed_adjacency(
                candidates
            )
        )

        utility_by_edge = {
            (
                candidate["i"],
                candidate["j"]
            ):
                candidate["utility"]

            for candidate
            in candidates
        }

        old_edges = (
            self._inter_edges(
                self.current_adj
            )
        )

        (
            new_adj,
            new_W,
            proposed_drift,
            actual_drift
        ) = self._bounded_rewire(
            proposed_adj,
            utility_by_edge
        )

        new_edges = (
            self._inter_edges(
                new_adj
            )
        )

        # ----------------------------------------------------
        # Final safety guarantees
        # ----------------------------------------------------
        if not self._is_connected(
            new_adj
        ):
            raise RuntimeError(
                "Final topology became disconnected."
            )

        self._validate_inter_degree(
            new_adj
        )

        self._validate_mixing_matrix(
            new_W
        )

        if (
            actual_drift
            >
            self.delta_max
            + 1e-8
        ):

            raise RuntimeError(
                "Bounded topology drift constraint violated."
            )

        # ----------------------------------------------------
        # Commit topology
        # ----------------------------------------------------
        self.current_adj = (
            new_adj
        )

        self.current_W = (
            new_W
        )

        # ----------------------------------------------------
        # Diagnostics for thesis/experiments
        # ----------------------------------------------------
        selected_candidates = []

        for edge in new_edges:

            for candidate in candidates:

                if (
                    candidate["i"],
                    candidate["j"]
                ) == edge:

                    selected_candidates.append(
                        candidate
                    )

                    break

        if selected_candidates:

            mean_selected_stat = float(
                np.mean(
                    [
                        candidate[
                            "statistical_score"
                        ]
                        for candidate
                        in selected_candidates
                    ]
                )
            )

            mean_selected_pairwise = float(
                np.mean(
                    [
                        candidate[
                            "pairwise_cosine"
                        ]
                        for candidate
                        in selected_candidates
                    ]
                )
            )

            mean_selected_delay = float(
                np.mean(
                    [
                        candidate[
                            "delay"
                        ]
                        for candidate
                        in selected_candidates
                    ]
                )
            )

            mean_selected_utility = float(
                np.mean(
                    [
                        candidate[
                            "utility"
                        ]
                        for candidate
                        in selected_candidates
                    ]
                )
            )

        else:

            mean_selected_stat = 0.0
            mean_selected_pairwise = 0.0
            mean_selected_delay = 0.0
            mean_selected_utility = 0.0

        self.last_metrics = {

            "round":
                round_num,

            "topology_updated":
                (
                    old_edges
                    != new_edges
                ),

            # Novelty 1 diagnostics
            "mean_selected_target_alignment":
                mean_selected_stat,

            "mean_selected_pairwise_cosine":
                mean_selected_pairwise,

            "ema_target_norms":
                {
                    str(clique_id):
                        float(
                            np.linalg.norm(
                                target
                            )
                        )

                    for clique_id, target
                    in self.target_ema.items()
                },

            # Novelty 2 diagnostics
            "p85_cutoff":
                float(
                    p85_cutoff
                ),

            "filtered_candidate_edges":
                int(
                    sum(
                        not candidate[
                            "allowed_by_p85"
                        ]
                        for candidate
                        in candidates
                    )
                ),

            "mean_selected_delay":
                mean_selected_delay,

            "mean_selected_utility":
                mean_selected_utility,

            # Novelty 3 diagnostics
            "proposed_drift":
                float(
                    proposed_drift
                ),

            "actual_drift":
                float(
                    actual_drift
                ),

            "delta_max":
                float(
                    self.delta_max
                ),

            # Physical topology diagnostics
            "inter_edge_count":
                len(
                    new_edges
                ),

            "edge_changes":
                len(
                    old_edges
                    .symmetric_difference(
                        new_edges
                    )
                ),

            "max_inter_degree":
                max(
                    self._inter_degree(
                        new_adj,
                        i
                    )
                    for i
                    in range(
                        self.num_clients
                    )
                ),

            "connected":
                self._is_connected(
                    new_adj
                )
        }

        return (
            self.current_W
        )

    # ========================================================
    # Public diagnostics helpers
    # ========================================================

    def get_current_adjacency(
        self
    ):
        return (
            self.current_adj.copy()
        )

    def get_diagnostics(
        self
    ):
        return dict(
            self.last_metrics
        )