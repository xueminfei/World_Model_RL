# P5 Pre-registered prediction

Timestamp: 2026-06-15 (written BEFORE running the bad-model / data-sweep experiment)

Setup I am about to run: train the world model on only **200** transitions (a
"bad" model) and on **2000** transitions (a "medium" model), keeping everything
else identical, and run Dyna-Q (K=10) with each on the training map, alongside
the good 10,000-transition model and the K=0 baseline, 3 seeds, 50,000 real
steps. Then a 50K-step success-rate bar chart over {200, 2000, 10000, K=0}.

## What I predict, and why

1. **The 200-transition bad model will HURT.** With so few samples the transition
   net cannot have learned the noisy next-state distribution; its softmax is close
   to wrong/diffuse even for the visited (s,a) pairs that imagination samples. The
   K=10 imagined updates therefore inject systematically wrong Bellman targets,
   and because there are 10x more imagined updates than real ones, I expect the
   bad-model curve to be **dragged below the K=0 baseline** for much of training,
   i.e. the model makes Dyna-Q *worse than no model at all*. It may still crawl up
   because the 1 real update per step is correct, but I expect it to be clearly
   slower than K=0 and possibly to **plateau below 100%** at 50K.

2. **The 2000-transition medium model will be close to the good model.** I expect
   it to recover most of the Dyna-Q speed-up (fast early rise, near the
   10,000-transition curve), perhaps marginally slower.

3. **Threshold, not smooth.** I predict the data-scaling is **not** linear: 2000
   already buys most of the benefit (≈ as good as 10000), while 200 falls off a
   cliff. So Figure 5 should look like {200: low, 2000 ≈ 10000: high}, suggesting
   a threshold above which more data barely helps.

4. **50K bar chart caveat.** Because this map is easy and K=0 alone reaches 100%
   by ~20K, at 50K the K=0, 2000 and 10000 bars may all be ~100%; the
   discriminating bar is **200**, which I predict will be the only one clearly
   below 100% (if the poisoning is strong enough to persist to 50K).

Mechanism (Lec.18/19/21): a Bellman backup `Q(s,a) <- r + gamma*max_a' Q(s',a')`
is only as good as the (s',r) it is fed; a bad model supplies wrong s', so the
backup moves Q toward a wrong fixed point. Repeated K times per real step, the
wrong targets dominate and the value estimate is biased away from V*.
