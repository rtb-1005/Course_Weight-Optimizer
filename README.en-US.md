# Course_Weight-Optimizer

## Background and Purpose

Some courses at Northeastern University (China) adopt a “weight-based course selection” mechanism: each student is given a fixed budget of weight points (e.g., 105 points), which can be allocated across the courses they wish to take. Each course has a minimum bid threshold (e.g., at least 5 points to be considered a valid participant) and a capacity limit. At the system deadline, allocation is computed in a single clearing step: if the number of valid bidders for a course exceeds its capacity, the seats are assigned to the students with the highest weights until the capacity is filled; the remaining students fail to obtain a seat even though they have spent weight points. Before the deadline, students may freely revise or withdraw their bids, but they cannot observe how many points other individuals have bid—only aggregate information such as the current number of bidders per course is visible.

The goal of this repository is the following: given (1) your own course preferences and candidate course set, and (2) the global state (cohort size, course capacities, and current/snapshot bidder counts), the code outputs a recommended **weight allocation vector** that aims to maximize the probability of successfully enrolling in your desired courses under the given budget and minimum-bid constraints. In addition, the system reports a **range of admission probabilities** (conservative / neutral / aggressive scenarios) to help you assess risk and robustness.

It is important to emphasize that, because individual bidding distributions are not publicly observable, the output of this project is a **strategy recommendation under an auditable proxy model**, rather than an exact prediction of the true winning probabilities.

## Usage

You need to prepare two JSON files.  
`desired_courses.json` describes which courses you want to take and the relative importance (preference strength) of each course.  
`global_state.json` describes the current global state, including cohort size, course capacities, and snapshot bidder counts.  

After running the program, it outputs a recommended bid for each target course and classifies each course into one of three categories:

- **SAFE (under-enrolled)**: Under all three scenarios (conservative / neutral / aggressive), the predicted final number of bidders does not exceed capacity. In this case, bidding only the minimum entry ticket is sufficient, and the admission probability is treated as 1 in the model.
- **COMP (competitive)**: In some scenarios, the course is predicted to reach or exceed capacity. Additional weight must be allocated to improve ranking competitiveness. The algorithm applies a “water-filling” allocation so that more budget is concentrated on courses with higher marginal benefit.
- **OUT (not entered)**: When the budget is limited or the marginal benefit is too low, the algorithm may recommend not bidding on certain courses (bid 0), in order to avoid diluting competitiveness on more important courses with minimum-bid costs.

The JSON formats are as follows.

> `desired_courses.json`

```json
{
  "grade_size": 126,
  "courses": [
    {"course_id": "COURSE_01", "capacity": 30,  "bidders": 35},
    {"course_id": "COURSE_02", "capacity": 30,  "bidders": 30},
    {"course_id": "COURSE_03", "capacity": 30,  "bidders": 19},
    {"course_id": "COURSE_04", "capacity": 30,  "bidders": 31},
    {"course_id": "COURSE_05", "capacity": 30,  "bidders": 35},
    {"course_id": "COURSE_06", "capacity": 30,  "bidders": 31},
    {"course_id": "COURSE_07", "capacity": 153, "bidders": 58},
    {"course_id": "COURSE_08", "capacity": 30,  "bidders": 58}
  ]
}
````

> `global_state.json`

```json
{
  "preferences": [
    {"course_id": "COURSE_01", "utility": 7},
    {"course_id": "COURSE_02", "utility": 10},
    {"course_id": "COURSE_03", "utility": 8},
    {"course_id": "COURSE_06", "utility": 6},
    {"course_id": "COURSE_04", "utility": 6}
  ],
  "budget": 105,
  "min_bid": 5
}
```

The output also reports, for each course, a probability interval under the three scenarios (conservative / neutral / aggressive). This interval reflects how your proxy success probability varies as key uncertainties—such as the average number of courses each student ultimately enters—change across scenarios. You should pay particular attention to which courses remain high-probability across all scenarios and which are highly sensitive to scenario changes, indicating a need for more concentrated bidding or fewer total course entries.

## Modeling Logic and Algorithmic Construction

**Auction model overview:**
The course selection process is modeled as a multi-prize, complete-information *all-pay* auction. Each course corresponds to a prize (with multiple seats, i.e., multiple prizes), and students distribute weight points across multiple courses, effectively participating in several contests simultaneously. In such a model, seats are awarded to the highest bidders, while all participants pay their bids regardless of success. This closely matches reality: students spend limited points to compete for courses, and once spent, points cannot be recovered even if the course is not obtained. When competition is intense (bidders exceed capacity), only the top-ranked bids secure seats.

**Multi-battlefield allocation strategy (objective):**
Students must allocate a limited budget across multiple desired courses, which is analogous to the classical Colonel Blotto game—players distribute resources across multiple battlefields to win local contests. Kovenock and Roberson (2015) generalize this class of games into the Lotto framework, analyzing equilibrium resource allocation under budget constraints. Borrowing this idea, we view each student as a player allocating weight across multiple “course battlefields” to maximize overall payoff (course satisfaction). This multi-prize auction and multi-battlefield resource allocation framework provides the game-theoretic foundation for the algorithm and better reflects real competitive behavior.

**Constraints:**
To align with NEU’s actual rules, we impose minimum-bid and budget constraints. Each course requires a minimum bid to participate, and each student has a fixed, non-transferable budget (e.g., 105 points). Students cannot exceed this budget nor acquire points from others. This prevents unrealistic strategies such as placing tiny bids on all courses and ensures the simulation adheres to NEU’s mechanism.

**Greedy treatment of under-enrolled courses (minimum bid only):**
To avoid inefficiency and strategic distortion, we adopt a “probability-1 admission” rule for under-enrolled courses. If a course’s demand does not reach capacity, any student meeting the minimum bid is assumed to be admitted with probability 1. In this case, only the minimum necessary bid is recommended, and the remaining budget is reserved for more competitive courses. This prevents wasting weight on uncontested courses and improves overall efficiency. It also discourages exaggerated bidding driven by fear rather than true preference, making bids more reflective of genuine priorities.

**Main modeling assumptions:**
To simplify the model while remaining faithful to the theoretical framework, we adopt the following assumptions:

* Students are rational decision-makers who allocate weights to maximize expected utility (total expected satisfaction from obtaining desired courses).
* Preferences are independent and private: a student’s utility from a course does not depend directly on whether other students obtain that course, and preferences do not change in response to others’ choices (no externalities). Under complete information, students form reasonable expectations about competition.
* Budgets are fixed and non-transferable. Each student can only use their 105 points in the current cycle; unused points cannot be carried over or traded.
* For under-enrolled courses, all students bid only the minimum required weight.
* Cross-enrollment effects from students of other majors entering or leaving general elective courses are ignored, as such flows are limited and secondary.

> *Here, “dependence on others’ choices” refers only to informal information exchange (e.g., WeChat, QQ, campus forums) about course quality, not to the visible aggregate enrollment counts provided by the system.*

## Model Formulation and Algorithmic Analysis

Under the above framework, let $w_{ij}$ denote the weight allocated by student $i$ to course $j$. Each course $j$ has capacity $c_j$ and a minimum bid threshold $b_{\min}$. At the deadline, if the number of valid bidders exceeds capacity, seats are assigned to the top $c_j$ bidders. Because individual bid distributions are unobservable, the exact probability of ranking within the top $c_j$ cannot be identified. We therefore use a **monotonic, differentiable, and interpretable proxy probability model** to capture the qualitative relationship “more weight helps, higher congestion hurts,” and to generate actionable recommendations.

### 1) Decision Variables and Institutional Constraints

For a target student, let $J$ denote the set of desired courses, with bids $b_j$ for each course. The constraints are:

$$
\sum_{j\in J} b_j = W,\qquad
b_j \in {0}\cup [b_{\min},+\infty),
$$

where $W$ is the budget (e.g., $W=105$ after evaluation points), and $b_{\min}$ is the minimum bid (e.g., $b_{\min}=5$). This expresses that the entire budget is allocated among courses and that a course is either not entered or entered with at least the minimum bid.

The maximum number of courses that can be meaningfully entered is:

$$
K_{\max}=\frac{W}{b_{\min}}.
$$

### 2) Duplication Intensity and Per-Course Budget Scale

Differences in cohort size can cause large differences in effective bid scales. The root cause is that global “bidder counts” are aggregated across courses, so students entering multiple courses are counted repeatedly. Let $P$ be cohort size, $N$ the total number of courses, and $m_j$ the snapshot number of valid bidders for course $j$. Define:

$$
M \triangleq \sum_{j=1}^{N} m_j,\qquad
\bar{s} \triangleq \frac{M}{P}.
$$

Here, $M$ is the total number of course entries, and $\bar{s}$ is the estimated average number of courses entered per student. Larger $\bar{s}$ implies stronger duplication and smaller effective per-course budgets. We clip this quantity:

$$
\bar{s}\leftarrow \mathrm{clip}(\bar{s},1,K_{\max}),
$$

and define the calibrated per-course budget scale:

$$
\mu \triangleq \frac{W}{\bar{s}}.
$$

This ensures that when $P$ changes, $\mu$ rescales automatically, preventing order-of-magnitude errors in recommended bids under otherwise similar enrollment snapshots.

### 3) Terminal Uncertainty and Scenario-Based Treatment of Under-Enrolled Courses

Because only snapshots are available, we extrapolate possible terminal enrollment using three scenarios:

$$
s^{(t)} = \mathrm{clip}(\lambda_t \bar{s},1,K_{\max}),
\quad \lambda_t\in{0.8,1.0,1.2},
\qquad
M^{*(t)} = P\cdot s^{(t)}.
$$

The predicted incremental entries are:

$$
\Delta^{(t)} = \max{0,; M^{*(t)} - M}.
$$

Without directional information, we distribute increments proportionally to current popularity:

$$
q_j=
\begin{cases}
\dfrac{m_j}{\sum_{k=1}^N m_k}, & \sum_{k=1}^N m_k>0,[6pt]
\dfrac{1}{N}, & \text{otherwise},
\end{cases}
\qquad
\hat{m}_j^{(t)} = \min{P,; m_j + \Delta^{(t)} q_j}.
$$

A course is classified as robustly under-enrolled if:

$$
\max_t \hat{m}*j^{(t)} \le c_j
\quad\Longrightarrow\quad
b_j=b*{\min},;; \Pr(\text{admission to }j)=1.
$$

This captures the key intuition: when a course remains under capacity even in the worst case, extra bidding yields no marginal benefit and budget should be conserved for competitive courses.

### 4) Competitive Courses: From Contest Success Functions to a Proxy Probability

In single-prize contests with complete information, a common **contest success function (CSF)** is:

$$
P_{ij} = \frac{w_{ij}}{w_{ij} + \sum_{k\neq i} w_{kj}}.
$$

While intuitive, NEU courses usually have multiple seats ($c_j>1$) and are awarded by rank, so the true probability is a complex ordering event. Instead of using the CSF directly, we define a smooth proxy based on congestion.

Define congestion under scenario $t$:

$$
\rho_j^{(t)} \triangleq \frac{\hat{m}_j^{(t)}}{c_j}.
$$

Map congestion to a difficulty scale:

$$
\alpha_j^{(t)}
\triangleq
\mu^{(t)}\cdot \ln\Big(\max\big(\rho_j^{(t)},,1+\delta\big)\Big),
\qquad
\mu^{(t)}=\frac{W}{s^{(t)}},
$$

where $\delta>0$ is a smoothing constant (e.g., $\delta=0.05$). The proxy success probability is then:

$$
\pi_j^{(t)}(b_j)\triangleq 1-\exp\Big(-\frac{b_j}{\alpha_j^{(t)}+\varepsilon}\Big),
\qquad \varepsilon>0.
$$

This function is increasing and differentiable in $b_j$, and reflects that higher congestion makes marginal improvements harder.

### 5) Expected Utility and Optimization Objective

Let $u_j$ denote the utility (importance) of course $j$. The expected utility under scenario $t$ is:

$$
\mathcal{U}^{(t)}(\mathbf{b})

\sum_{j\in J_{\text{COMP}}} u_j,\pi_j^{(t)}(b_j)
+
\sum_{j\in J_{\text{SAFE}}} u_j.
$$

In practice, the neutral scenario is used to generate a concrete bid vector, while all three scenarios are reported to show robustness:

$$
\mathbf{b}^{*} \approx \arg\max_{\mathbf{b}} \mathcal{U}^{(\text{neutral})}(\mathbf{b})
\quad \text{s.t. budget (105) and minimum-bid (5) constraints.}
$$

### 6) Water-Filling Allocation and One-Dimensional Search

After fixing the set of competitive courses $S\subseteq J_{\text{COMP}}$, the remaining budget is:

$$
W' \triangleq W - |J_{\text{SAFE}}|,b_{\min},
\qquad
\sum_{j\in S} b_j = W',
\qquad b_j\ge b_{\min}.
$$

The KKT conditions yield a water-filling form: there exists $\nu>0$ such that

$$
b_j^{*}
=
b_{\min}+\max\Big\{0,\;(\alpha_j+\varepsilon)\ln\frac{u_j}{\nu(\alpha_j+\varepsilon)}-b_{\min}\Big\},
\quad j\in S,
$$

with $\nu$ determined by a one-dimensional binary search so that $\sum_{j\in S} b_j^{*}=W'$. Courses with low marginal benefit receive only the minimum bid or are excluded entirely.

In summary, this project combines insights from mechanism design and contest games to construct an interpretable, reproducible, and executable recommendation system. Rather than predicting exact admission probabilities, it provides rational decision support for allocating limited bidding weights under uncertainty and information constraints.

## References

> [1] Budish, E., Cachon, G. P., Kessler, J. B., & Othman, A. (Course Match and point-based course allocation).\
> [2] Barut, Y., & Kovenock, D. (1998). *The Symmetric Multiple Prize All-Pay Auction with Complete Information*.\
> [3] Kovenock, D., & Roberson, B. (General Lotto / Colonel Blotto generalizations).\
> [4] Sönmez, T., & Ünver, M. U. (2010). *Combinatorial assignment and matching*.\
> *Full bibliographic entries are provided in `ref.bib`.*

Finally, best wishes to everyone in selecting their desired courses!

