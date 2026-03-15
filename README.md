# When More is Less: Simple Entropy Regularization Outperforms Sophisticated RLHF Methods in Non-Stationary Policy Gradient Learning

## Abstract

Policy gradient methods remain a cornerstone of reinforcement learning, yet they are known to struggle when environments contain stochastic noise, sudden reversals in constraints, and gradual shifts in reward preferences. To study this fragility systematically, we introduce the Noisy REINFORCE Disorder benchmark---a compact MDP that combines randomised budget contraction, post-reversal penalties, and a linear preference drift toward cheaper actions. Five automatically computed symptoms (impulsivity, credit-assignment failure, jumpy learning, reversal rigidity, and drift adaptation) are aggregated into a composite health score, while an iso-dose scale quantifies algorithmic intervention strength.

We compare nine treatment arms ranging from raw REINFORCE (T0) through moving-average baselines, entropy regularization with temperature annealing (T2), actor-critic variants, full PPO (T5), and three recent RLHF-inspired critic-free methods: leave-one-out (RLOO), group-relative policy optimization (GRPO), and global running normalization (REINFORCE++). Across eight random seeds and 600 episodes per arm, T2 achieved the highest mean health score (0.650 ± 0.104) and the highest mean return (37.8), together with near-zero rigidity and drift symptoms. REINFORCE++ (T8) provided a competitive second option at higher dose. In contrast, arms relying on learned critics or clipping consistently produced negative returns and maximal rigidity despite strong credit-assignment statistics on paper.

The health-versus-dose relationship formed a clear inverted U, suggesting that beyond a modest intervention level additional stabilizers can induce rigidity rather than robustness. These findings do not claim universal superiority for simple methods, but they highlight the value of starting with light entropy regularization and escalating complexity only when domain-specific ablation confirms a need. The pipeline and full results are publicly released to support further testing in continuous control, LLM alignment, and multi-agent settings.

## 1. Introduction {#introduction}

Policy-gradient methods have been central to reinforcement learning for many years, especially in problems that require sequential decisions. A key starting point is REINFORCE, introduced by Williams \[1\], which offers a direct and model-free way to improve a stochastic policy from sampled returns. Its appeal lies in its simplicity: the policy is adjusted in the direction of better expected reward without requiring an explicit model of the environment. At the same time, Williams \[1\] makes clear that these methods face an important practical limitation, namely the high variance of their gradient estimates. That weakness becomes especially consequential when rewards are noisy, when the environment changes abruptly, or when the relative value of actions shifts over time.

These difficulties matter because many practical settings are not stable. An agent may learn under one set of conditions and then be forced to operate under another, with little warning. In such cases, reward noise can magnify already unstable updates, while delayed or changing reward structure makes it harder to identify which actions should be reinforced. This is not only a classic control problem. It also bears on contemporary alignment settings, where reinforcement learning from human feedback has become a prominent approach for improving large language models. As Ahmadian et al. \[2\] note, RLHF has largely treated PPO as the standard optimization method, even though it comes with substantial computational cost and sensitive tuning requirements. Their argument for returning to simpler REINFORCE-style methods is therefore relevant not only for efficiency, but also for thinking about robustness when training conditions are imperfect or change over time.

1.1 The fragility of policy-gradient methods in the wild (noise, reversal, drift)

The main source of fragility is the variance in Monte Carlo policy-gradient estimates. In Williams \[1\], REINFORCE algorithms are presented as gradient-following procedures for stochastic networks, but the paper also identifies high variance as a central challenge. When rewards are noisy, this problem intensifies because updates become more sensitive to random fluctuations rather than consistent signal. If the environment is also non-stationary, the issue becomes harder still: policies are not merely learning from noisy outcomes, but from outcomes whose meaning may already be changing.

This becomes especially problematic when there are reversals or gradual drift in what counts as a good action. A sudden change can make previously useful actions poor choices, while a slower shift can alter action rankings without any clear boundary between old and new conditions. Under these circumstances, policy-gradient methods can either react too strongly to noise or remain tied to outdated behavior. The result is often unstable learning, poor adaptation, and large drops in return after the environment changes.

A related concern appears in modern alignment pipelines. Ahmadian et al. \[2\] describe RLHF as a setting in which PPO has become standard despite its complexity, and they question whether that complexity is always justified. Their motivation suggests that when optimization is expensive and sensitive, additional machinery may not always solve the underlying difficulty. In settings where the reward signal itself is imperfect or evolving, complexity can sometimes mask rather than resolve the problem.

1.2 Why \"more bells and whistles\" often backfires: critics, clipping, batching, and PPO

A lot of the later work in policy optimization was driven by one core issue identified early on: variance. To make training more stable and sample-efficient, researchers introduced things like learned critics, normalization, clipping, and batch-based objectives. In RLHF, PPO became the standard approach largely because it seemed to address those concerns well. But Ahmadian et al. \[2\] argue that, for large language models trained from human feedback, many of the original reasons for using PPO are not nearly as convincing. They point especially to its computational cost and sensitivity to hyperparameters, and suggest that simpler methods may actually be a better fit.

Their main point is not just that simpler methods are cheaper or easier to run. It is that they can sometimes perform better, too. Ahmadian et al. \[2\] revisit the RL formulation behind RLHF and show that several ingredients commonly tied to PPO may not actually be necessary in this setting. In their results, simpler REINFORCE-style methods outperform both PPO and newer so-called \"RL-free\" approaches like DPO and RAFT. The broader lesson is that adding more optimization machinery does not automatically lead to better outcomes. Sometimes it just adds rigidity, complexity, and extra tuning work without providing enough benefit in return.

Seen in this light, the appeal of returning to REINFORCE is both conceptual and practical. Williams \[1\] established a broad class of reinforcement-learning algorithms that follow the gradient of expected reinforcement without explicitly computing or storing full gradient estimates. Ahmadian et al. \[2\] show that this basic idea remains highly relevant in modern LLM alignment. Taken together, the two papers suggest that simplicity is not merely a historical baseline; it may still offer a strong and sometimes preferable foundation for policy optimization.

1.3 Contributions

This section builds on two complementary sources. Williams \[1\] provides the original theoretical foundation for REINFORCE, describing a general class of associative reinforcement-learning algorithms for connectionist networks with stochastic units. These algorithms are shown to adjust weights in a direction aligned with the gradient of expected reinforcement, both for immediate-reinforcement problems and for certain limited delayed-reinforcement settings. The paper also presents concrete examples, discusses links to earlier methods, shows how REINFORCE can be integrated with backpropagation, and closes with discussion of limiting behavior and possible directions for more powerful reinforcement-learning algorithms.

Ahmadian et al. \[2\] revisit that tradition in the much newer setting of RLHF for large language models. Their contribution is to argue that PPO, despite becoming the dominant method in this area, is not necessarily the most appropriate choice. They contend that PPO\'s high computational cost and sensitivity to tuning make it unnecessarily burdensome for RLHF, and they show that simpler REINFORCE-style optimization variants can preserve or even improve performance. In their account, many elements of PPO are not essential in the RLHF setting, and careful adaptation of simpler online RL methods can outperform both PPO and recently proposed alternatives marketed as RL-free.

Taken together, these works support a clear line of argument for this section: the basic gradient-following logic of REINFORCE remains important, and its value may be even greater in modern applications where optimization overhead matters. Williams \[1\] gives the foundational mechanism; Ahmadian et al. \[2\] show that, in LLM alignment, returning to that mechanism can be both efficient and empirically strong.

1.4 Paper roadmap

The discussion that follows is organized around this connection between foundational and contemporary work. It begins from the original REINFORCE framework introduced by Williams \[1\], then turns to the recent re-evaluation of REINFORCE-style methods for RLHF in Ahmadian et al. \[2\]. The aim is to show not only how policy-gradient methods developed in early connectionist reinforcement learning, but also why those ideas have regained relevance in present-day large language model alignment.

More specifically, the next parts trace a simple argument. First, REINFORCE offered a direct way to follow the gradient of expected reinforcement in stochastic systems \[1\]. Second, later optimization practice often moved toward more elaborate methods such as PPO. Third, recent evidence from RLHF indicates that this added complexity is not always necessary and may, in some cases, be inferior to well-adapted REINFORCE-style optimization \[2\]. Framed this way, the broader lesson is straightforward: when the objective is effective policy improvement, simplicity can remain a serious advantage rather than a limitation.

## 2. Related Work {#related-work}

Policy-gradient research has expanded in several directions since the early REINFORCE work. Much of that effort has centered on three recurring problems: unstable updates, changing reward conditions, and the practical demands of newer alignment settings. The literature relevant to this study can be grouped into four broad areas: classical attempts to reduce variance in REINFORCE, actor-critic and PPO-based approaches, recent simplifications proposed for RLHF, and studies of drift and reversal in learning systems. Taken together, these strands show a field that has often responded to instability by adding new layers of optimization, even as some recent work has begun to argue for simpler alternatives.

2.1 Variance reduction in REINFORCE (baselines, advantage estimation, entropy)

The starting point remains REINFORCE. Williams \[1\] described a general class of associative reinforcement-learning algorithms for connectionist networks with stochastic units and showed that their weight updates move in a direction aligned with the gradient of expected reinforcement. This holds for immediate-reinforcement tasks and for some limited forms of delayed reinforcement as well. One of the attractions of the method is that it does not require explicit gradient estimates or storage of the information needed to reconstruct them later. Even so, the practical weakness of Monte Carlo-style updates has long been their high variance.

That concern led to a series of relatively simple fixes. One of the earliest and most common was subtracting a baseline from returns in order to reduce variance without changing the expected gradient \[3\]. Over time, this simple idea grew into moving-average baselines and more general ways to estimate advantage, such as those based on learned value functions. Another strategy that is related but different is to add entropy regularization to the objective to encourage continued exploration. This can help stop early collapse to overly certain action choices \[4\]. In practice, these additions are appealing because they are relatively light and often make learning easier in noisy environments without putting too much strain on the computer.

Another common adjustment is to control exploration more deliberately over training. Temperature annealing in softmax policies serves that purpose by gradually changing how sharply the policy favors higher-valued actions, and this is often used alongside entropy-based objectives \[5\]. These methods do not remove the underlying variance problem, but they represent a long-standing attempt to make REINFORCE-style learning more stable without abandoning its basic simplicity.

2.2 Actor-Critic and PPO in non-stationary environments

A more substantial response to variance came from actor-critic methods. Rather than relying entirely on Monte Carlo returns, these approaches train a separate critic to estimate values or advantages, which allows policy updates to use lower-variance targets \[3\]. PPO extends this line of work by combining the actor-critic structure with a clipped surrogate objective, gradient clipping, and mini-batch updates designed to keep policy changes from becoming too large at once \[6\]. These methods have been influential because they offered a practical recipe for more stable optimization in many reinforcement-learning settings.

Their success, however, does not settle the question of how well they cope with change. In environments that shift over time, a critic can become stale, advantage estimates may reflect yesterday\'s reward structure rather than today\'s, and clipping can restrict the policy movement needed to discover a newly better course of action. Ouyang et al. \[7\] point to related difficulties in alignment settings, where systems trained with human feedback can fail to keep up with updated preferences unless additional adaptation occurs. Although techniques such as normalization and adaptive scheduling have been used to soften these problems, the evidence remains mixed once the environment becomes clearly non-stationary.

2.3 RLHF-specific techniques (RLOO, GRPO, global normalisation) and their reported robustness

Recent work in RLHF has reopened the case for simpler policy-gradient methods. Ahmadian et al. \[2\] argue that PPO became the standard method for the RL stage of alignment largely through established practice, not because it is always the most suitable tool. In their account, PPO brings both high computational cost and sensitive hyperparameter tuning. They therefore revisit REINFORCE-style optimization with simplicity as the guiding principle and report that many parts of PPO are unnecessary in the RLHF setting. Their main claim is stronger than a plea for efficiency: they find that simpler REINFORCE-style variants can outperform PPO as well as newer \"RL-free\" approaches such as DPO and RAFT \[2\].

This argument has encouraged renewed interest in methods that avoid a learned critic or heavy optimization stack. One prominent example is the leave-one-out approach associated with REINFORCE-style RLHF, where the baseline is derived from multiple on-policy samples rather than a separate value model \[2\]. Other recent methods, including group-relative schemes such as GRPO, similarly try to stabilize updates through normalization within groups rather than through a fully trained critic \[8\]. Global running normalization has also appeared as a lighter alternative to batch-specific or learned baselines. These approaches are usually presented as practical and robust, especially when compute or memory is limited, but most of that discussion has focused on alignment tasks rather than sharply defined reversal or drift settings.

2.4 Studies on reward drift and reversal learning (contextual bandits, continual RL)

Questions of drift and reversal have been explored more directly in neighboring areas of reinforcement learning. In contextual bandits, methods that use forgetting mechanisms or sliding windows have been shown to track changing rewards better than stationary approaches \[9\]. In continual reinforcement learning more broadly, abrupt or gradual environmental change is often associated with two familiar failures: rigidity after reversal and forgetting after adaptation \[10\]. These issues matter for alignment as well. Ouyang et al. \[7\] note that when preferences shift, alignment pipelines may no longer reflect the updated target unless retraining or continual adaptation is brought in.

There is also growing interest in adjusting exploration itself in response to change. Wang et al. \[11\], for example, examine entropy scheduling for non-stationary reinforcement learning, suggesting that exploration may need to become sensitive to variation rather than being fixed in advance. This line of work reinforces a broader point: once reward conditions stop being stable, the difficulty is not only learning a good policy, but also retaining enough flexibility to notice when the old policy has stopped working.

## 3. The Noisy REINFORCE Disorder Framework {#the-noisy-reinforce-disorder-framework}

To study how policy-gradient methods behave when learning becomes fragile, this framework was built as a compact diagnostic setting. Its purpose is to keep the task small and controlled while bringing together the kinds of instability that often make policy optimization difficult in practice. The design combines three pressures in the same environment: noisy rewards, abrupt change, and shifting preferences. Around that core, the framework adds a symptom-scoring scheme, a simple way of comparing intervention strength across algorithms, and a fixed training pipeline so that results can be compared on the same basis. The overall idea follows the long-standing concern in reinforcement learning with how policies adapt under uncertainty, while remaining close to the gradient-following tradition established by REINFORCE \[1\].

3.1 MDP formulation (budget planning with randomised reversal and preference drift)

The environment is a finite-horizon episodic Markov decision process with a horizon of 10 steps. At each step, the state is represented by three values: the normalized day within the episode, the fraction of the initial budget already spent, and the remaining budget normalized by its initial value. The state is deliberately simple and fully observable. That choice is meant to keep attention on the learning rule rather than on issues such as hidden state or complicated transitions.

The agent chooses among four actions that correspond to different spending levels. These are luxury, with cost 120; standard, with cost 60; economy, with cost 25; and skip, with cost 0. Before any environmental change, their base rewards are 15, 10, 5, and 1, respectively. In other words, the more expensive choices are initially the most rewarding. To make the task less clean and more realistic, both costs and rewards are perturbed by independent Gaussian noise. The standard deviation is 5 for costs and 12 for rewards.

A reversal occurs partway through the episode. The reversal step is drawn uniformly from the set {3, 4, 5, 6, 7}, so the timing changes from episode to episode and is not known in advance by the agent. When that point is reached, the available budget drops sharply from 1500 to 400. From then on, taking either the luxury or standard action also incurs an additional per-step penalty of 3. If total spending by the end of the episode exceeds the final budget, the agent receives a terminal penalty proportional to the overspend, scaled by the number of steps.

The framework also includes preference drift after the reversal. A linear interpolation parameter increases from 0 to 1 over the remaining steps and gradually shifts the reward vector away from its original form toward a new economy-favoring reward pattern: \[2, 5, 12, 8\]. As this happens, the action that looked relatively unattractive early in the episode becomes the most valuable one later on. This is meant to capture a gradual change in what counts as a good choice, rather than a clean switch announced to the agent. In combination, the randomized reversal, the post-reversal penalties, and the drifting reward ranking create a compact environment that contains both sudden shocks and slower forms of change. That makes it suitable for testing how well gradient-following methods handle conditions that move away from the stable settings in which they are often first introduced \[1\].

3.2 Symptom scoring (Impulsivity, Credit Assignment, Jumpy Learning, Reversal Rigidity, Drift Adaptation) and composite Health metric

Performance in the framework is summarized through five symptom scores. Each score is scaled to the interval from 0 to 1, with lower values indicating better behavior. The aim is not only to ask whether a method gets higher return, but also to identify the particular way in which it struggles.

The first symptom is impulsivity. This is computed as the average standard deviation of the action sequence within each episode, divided by the theoretical maximum standard deviation for the four available actions. A high score suggests that the policy continues to behave erratically and keeps sampling broadly even after enough experience has accumulated to support more stable choices.

The second symptom is credit assignment failure. This is measured as one minus the absolute correlation between the standard deviation of actions in the first half of an episode and the final return for that episode. If early variation in behavior is meaningfully related to later outcomes, that correlation should be strong. If it is weak, the score rises, indicating that the policy is not effectively linking early decisions with downstream consequences.

The third symptom is jumpy learning. This is defined as the coefficient of variation of returns during the final quarter of training. When this value remains high late in training, it points to continuing instability rather than steady settling.

The fourth symptom is reversal rigidity. This is based on the relative drop in average per-step reward before versus after the reversal point, aggregated across episodes and clipped at 1. A large drop means that the policy has failed to adapt once the environment changes and continues to behave as if earlier conditions still held.

The fifth symptom is drift adaptation. This measure focuses on whether the policy shifts toward the newly favorable lower-cost actions after the reversal. It is computed from the fraction of post-reversal actions in late training that favor the cheap choices, meaning actions at or above 2. The symptom score is one minus that fraction. As a result, the score rises when the agent continues to prefer expensive actions even after the reward ranking has changed.

These five symptoms are combined into a composite health score by averaging them and subtracting the result from one. Scores are reported to three decimal places. Because all five measures are calculated from the same training trajectories, the health score is meant to give a single summary of learning quality while still reflecting several distinct forms of failure. It is based on the full training process rather than on a separate held-out evaluation phase, so it describes how the method actually learned under the framework\'s conditions.

3.3 Iso-dose normalisation (intervention intensity scale 0--145)

To compare algorithms that differ not just in results but also in how much extra machinery they add to basic REINFORCE, the framework uses an iso-dose scale. This is a simple intervention-strength measure built by assigning a fixed point value to each added feature. Reducing reward noise from the untreated level of 20 to 6 contributes up to 21 points. A moving-average baseline contributes 15 points. Entropy regularization contributes 15, gradient clipping 10, advantage normalization 10, a learned critic 25, reward normalization 8, PPO-style clipping 20, temperature annealing 8, adaptive learning-rate decay 5, and batching beyond a single episode 8.

Under this scheme, untreated REINFORCE has a dose of 0, while the full PPO arm reaches roughly 145. After results are aggregated across seeds, the raw dose values are divided by the maximum observed value so they lie on a unit interval. This makes it possible to compare health against intervention intensity and to ask a simple question: as more stabilization features are added, does performance continue to improve, or do returns begin to flatten or even worsen?

The measure is intentionally rough. It is not presented as a formal estimate of computational cost or theoretical importance. Its purpose is simply to provide a clear and reproducible way of placing methods on a common scale, somewhat like a dose-response comparison in other empirical settings.

3.4 Training pipeline (seeds, episodes, GPU, JSON logging, adaptive LR, reward-norm warmup, entropy annealing)

All experiments use eight random seeds so that results are not tied to a single initialization or trajectory sample. Each treatment is trained for 600 episodes. Runs are performed on a single GPU when CUDA is available, or on CPU otherwise. The model architecture is fixed across conditions: both policy and value networks, when a value head is used, have two hidden layers of 64 and 32 units with LayerNorm and tanh activations. Weights are initialized orthogonally. The optimizer is Adam with a base learning rate of 0.003.

Several training features are shared across treatment arms so that comparisons reflect the intended algorithmic differences rather than unrelated implementation choices. When reward normalization is active, statistics are collected during the first 50 episodes as a warm-up period, after which rewards are scaled by their running standard deviation. When entropy regularization is used, the entropy coefficient begins at either 0.05 or 0.03 and decays multiplicatively by 0.995 or 0.997 per episode until it reaches a floor of 0.001. Temperature annealing follows a similar pattern and decays from its initial multiplier toward 0.3. Adaptive learning-rate decay is triggered when the coefficient of variation over the most recent 50 episode returns exceeds 0.8; in that case, the learning rate is multiplied by 0.995 until it reaches a lower bound of 1e-5. These mechanisms are turned on or off depending on the treatment being tested.

Experience is collected in episode buffers of size 1 or 4, depending on the treatment arm, and each buffer is followed by a single policy update. Full logs are written to JSON after every run, including per-seed trajectories and the symptom breakdowns. The framework also produces learning curves, radar plots, health-versus-dose plots, and symptom-improvement heatmaps automatically, although those visual outputs are supplementary rather than essential to the main comparison.

The main purpose of this pipeline is consistency. Network structure, number of seeds, training length, and most hyperparameters are held fixed, while only the treatment-specific settings change. That way, differences in behavior can be attributed more directly to the learning method itself. In a setting designed to expose how gradient-following approaches respond to noise, reversal, and drift, that consistency is essential for a fair comparison \[1\].

## 4. The Nine Treatment Arms {#the-nine-treatment-arms}

We tested nine policy-gradient variants, each built from the same REINFORCE starting point but differing in how much extra machinery was added. The sequence was arranged from the simplest form of update to more elaborate variants that use baselines, entropy terms, critics, batching, clipping, or critic-free alternatives. The first part of the sequence follows a familiar progression from raw REINFORCE toward PPO-style training. The later arms draw on more recent RLHF work that questions whether a learned critic and full PPO setup are always necessary \[2\]. Across all arms, the architecture and most training settings were kept fixed so that differences in performance could be traced to the update design itself rather than to broader implementation changes.

4.1 T0--T5 (untreated → full PPO, retained from v2.1)

T0 serves as the untreated condition. It uses raw REINFORCE with Monte Carlo returns and no baseline, no regularization, and no clipping. Reward noise is set deliberately high, with a standard deviation of 20.0, and the initial temperature multiplier is 1.5. The policy is updated after every episode. In practical terms, this arm shows what happens when the original gradient-following logic is used with almost no stabilizing additions. That makes it a natural reference point for the rest of the framework, especially given that REINFORCE was originally presented as a direct way to move along the gradient of expected reinforcement without explicitly computing or storing full gradient estimates \[1\].

T1 adds a simple moving-average baseline computed from recent episode returns. Reward noise is reduced to 12.0, and the temperature multiplier is lowered to 1.0. This is the first step away from untreated REINFORCE and reflects the classic use of a baseline as a lightweight way to reduce variance.

T2 retains that simpler structure but adds entropy regularization and temperature annealing. The entropy coefficient starts at 0.05 and decays multiplicatively by 0.995 per episode until it reaches a floor of 0.001. Temperature follows its own annealing schedule, moving from 1.0 toward 0.3. Reward noise remains at 12.0. This arm is meant to test whether a modest encouragement toward exploration can improve learning without introducing a critic or PPO-style objective.

T3 marks a larger change. It introduces a learned value function for generalized advantage estimation, gradient clipping at 0.5, per-batch advantage normalization, adaptive learning-rate decay triggered when the coefficient of variation of recent returns exceeds 0.8, and a further reduction of reward noise to 8.0. A critic is trained alongside the policy. This is the point at which the treatment sequence moves away from simple REINFORCE-style variance control and toward a more conventional actor-critic structure.

T4 extends T3 by adding reward normalization and batching four episodes per update. Reward-normalization statistics are gathered during the first 50 episodes before scaling begins. Reward noise falls to 6.0. The entropy coefficient now starts at 0.03 and decays by 0.997. Batch size is fixed at 4. The main shift here is that the system becomes not only critic-based but also more buffered and normalized.

T5 is the full PPO condition. It takes T4 and adds the clipped surrogate objective associated with proximal policy optimization \[6\]. It uses four PPO epochs per batch update, a clip range of 0.2, and keeps the critic, reward normalization, gradient clipping, entropy term, and adaptive decay from the previous arm. Reward noise remains at 6.0. Together, T0 through T5 trace a familiar escalation path: raw REINFORCE, baseline subtraction, entropy support, critic-based estimation, batching, and finally PPO-style constraints.

4.2 New RLHF-inspired arms: T6 RLOO (§13), T7 GRPO (§14), T8 REINFORCE++ (§15)

The final three arms were added to reflect the newer RLHF literature, where simpler critic-free approaches have attracted attention because full actor-critic pipelines are expensive and not always clearly better. Ahmadian et al. \[2\] argue directly that many components of PPO are unnecessary in RLHF and that simpler REINFORCE-style variants can outperform PPO as well as \"RL-free\" alternatives such as DPO and RAFT. These arms were designed to test that broader idea in the present framework.

T6 implements a leave-one-out baseline across the four concurrent rollouts in a batch. For each episode, the advantage is computed by subtracting the average return of the other three episodes. No value network is trained. Gradient clipping, entropy regularization, temperature annealing, and reward noise of 6.0 are retained, and batch size remains 4. This arm follows the logic behind RLOO-style optimization, which replaces the learned critic with a batch-based reference signal and is motivated in part by efficiency concerns in RLHF \[2\].

T7 uses group-relative normalization within the batch. Episode returns are normalized by the batch mean and standard deviation before the PPO clipped surrogate is applied. As in T6, there is no separate critic. The clipping, entropy schedule, annealing, and low-noise setting are the same as in T6. This arm was included because group-relative methods have been presented as a way to stabilize training without relying on a learned value function, and GRPO has become associated with recent reasoning-focused training pipelines \[8\].

T8 uses a global running normalization of returns. Instead of comparing only within a batch, it maintains a running mean and standard deviation over a 500-episode window and uses those values to normalize each return. This creates a critic-free advantage signal that evolves throughout training. Entropy regularization, gradient clipping, batch-level advantage normalization, temperature annealing, adaptive learning-rate decay, batch size 4, and reward noise of 6.0 are all active in this arm. The point of including T8 is to test whether a longer-horizon normalization scheme can provide a stable alternative to both a learned critic and a purely batch-local baseline.

Taken together, T6 to T8 ask whether the move back toward simpler REINFORCE-style optimization, which has been argued for in RLHF \[2\], still holds up when learning is stressed by reversal and drift rather than only by ordinary optimization cost.

4.3 Unified baseline system and per-arm hyper-parameters (noise levels, batch size, etc.)

To keep the comparison clean, all nine arms were implemented within the same training structure. Advantage computation was selected at runtime through a single baseline-type setting. The available options were none for raw returns, moving average for the earlier baseline-based arms, leave-one-out for T6, group-relative for T7, and global running for T8. This meant that the overall loop did not have to be rewritten for each arm. Only the logic used to form the policy signal changed.

The other main hyper-parameters were also organized to reflect the treatment progression rather than being separately tuned for each arm. Reward noise decreases in steps from 20.0 in T0 to 6.0 in T4 through T8. Batch size is 1 for T0 through T3 and 4 for T4 through T8. Entropy starts at 0.05 in T2 and T3, then 0.03 in T4 through T8, with decay rates of 0.995 or 0.997 depending on the arm. Temperature annealing is turned off only for T0 and T1. Adaptive learning-rate decay and reward-normalization warm-up are enabled only in the heavier conditions where they are part of the intended design. These settings were fixed before the comparison and not retuned per seed. That choice matters because it makes the observed differences easier to interpret as consequences of the treatment itself rather than as artifacts of arm-specific optimization.

4.4 Implementation details (networks, initialisation, optimisers)

The policy network and, where relevant, the value network use the same basic structure across all arms. Each has two hidden layers of 64 and 32 units, with LayerNorm and tanh activations. Policy outputs are passed through a softmax, with optional temperature scaling depending on the arm. Parameters are initialized orthogonally, with gain 0.01 for the policy head and 1.0 for the value head \[3\]. Optimization is done with Adam at a base learning rate of 0.003 using standard beta values. When gradient clipping is active, the norm is capped at 0.5.

All runs are carried out on a single GPU when available and otherwise on CPU. Seeding is deterministic for each seed-arm pair. Episode trajectories are stored until the required batch size is reached, after which the policy update and, where applicable, the critic update are performed in a single backward pass. Exact settings, including decay schedules and noise levels, are recorded in the codebase and JSON logs for reproducibility.

By holding the architecture, initialization, and optimizer constant while varying only the treatment-specific mechanisms, the nine-arm design provides a direct test of how baseline choice, regularization, clipping, and critic-free alternatives behave under the same noisy and shifting conditions. In that sense, the section remains anchored in the basic REINFORCE idea described by Williams \[1\], while also engaging the more recent argument from RLHF that simpler variants may sometimes be better suited than heavier PPO-style systems \[2\].

## 5. Experimental Setup {#experimental-setup}

All experiments were run under one common protocol so that the nine treatment arms could be compared on the same terms. The setup was kept intentionally straightforward. The aim was not to build the largest possible benchmark, but to create a controlled setting in which differences between methods would be easier to interpret. That choice is in keeping with the broader logic of REINFORCE-style analysis, where the behavior of the update rule itself is easier to study when the surrounding conditions are held steady \[1\].

5.1 Configuration (8 seeds × 600 episodes × 9 arms)

Each treatment arm was tested with eight independent random seeds. For every seed-arm pair, training ran for 600 episodes, and each episode lasted 10 steps. Across the full design, this produced 43,200 episode-level runs in total, coming from 8 seeds, 600 episodes, and 9 treatment arms. Training used a single GPU when CUDA was available and otherwise defaulted to CPU. The same network architecture, optimizer settings, and seed initialization procedure were used throughout so that the main source of variation would be the algorithmic differences between arms.

The choice of eight seeds reflected a compromise between reliability and cost. Early checks with fewer seeds showed that some arms could vary substantially, especially those more exposed to the timing of reversal. The choice of 600 episodes served a similar purpose. It gave most treatments enough time to show a stable late-stage pattern without making the full experiment impractically large. Other shared settings were also held fixed across the arms, including a discount factor of 0.99, a GAE lambda of 0.95, and a base learning rate of 0.003. Exceptions were made only where an arm\'s definition required them, such as changes in batch size or entropy scheduling.

5.2 Evaluation metrics (Health ± std, mean return, per-symptom scores, late-training stability)

The main summary measure was the composite health score introduced earlier. For each treatment arm, the reported value is the mean health score across the eight seeds, together with its standard deviation. Mean episode return over the full training run was also recorded, since the health score alone does not fully describe how much reward the policy actually collected.

The five individual symptom scores were also retained: impulsivity, credit assignment, jumpy learning, reversal rigidity, and drift adaptation. These were included because two methods can produce similar overall health values while failing in different ways. Looking at the symptom breakdown makes it possible to see whether a treatment mainly improves stability, adaptation, or some narrower part of behavior.

Late-stage behavior was examined through the return distribution in the final quarter of training, which corresponds to the last 150 episodes. This helps separate methods that truly settle into stable performance from those that remain erratic or deteriorate after reversal and drift. Iso-dose values were tracked as well, so that outcomes could be read against intervention intensity rather than only against arm labels.

5.3 Reproducibility (seeds, JSON output, public code)

Reproducibility was treated as a basic requirement of the setup. At the start of each seed-arm run, the random seed was fixed for both PyTorch and NumPy. After training, all trajectories, symptom values, and aggregate results were written to structured JSON files. These logs include per-seed returns, action traces, and symptom-level breakdowns, which makes it possible to verify the reported results or reanalyze them later in more detail.

The full codebase is described as containing the MDP definition, treatment implementations, symptom-scoring functions, and plotting routines. Configuration is handled through a single dataclass and can be changed through command-line arguments. This structure was meant to make the benchmark easy to reproduce exactly and easy to extend with new environments or additional treatment arms.

Overall, the design of the experiment was meant to reduce avoidable variation. By fixing the shared parts of training and changing only the treatment logic itself, the comparison stays focused on the question at the center of the study: how different forms of policy-gradient intervention behave under noise, reversal, and drift.

## 6. Results {#results}

Across the eight random seeds, the experiments produced a fairly consistent pattern. The aggregated results are reported in Table 4, while paired Wilcoxon signed-rank tests for health-score differences are shown in Tables 1 and 2. Table 3 gives 95% confidence intervals for the mean health scores using the t-distribution. Taken together, the results point in the same direction: lighter interventions generally produced better performance, especially once reversal and preference drift were introduced. Several of the heavier treatments, including those with critics and stronger clipping, performed worse than the untreated baseline.

6.1 Overall ranking: T2 Entropy (AMP) achieves the highest health and return, with near-zero rigidity & drift

The strongest overall performer was T2, the Entropy arm (Table 1). It reached a mean health score of 0.650 with a standard deviation of 0.104, and it also produced the highest mean return, 37.8. This was clearly better than the untreated T0 arm, which achieved a health score of 0.383 with a standard deviation of 0.072 and a mean return of 23.3. The improvement in health from T0 to T2 was statistically significant in the paired Wilcoxon test, with a change of 0.267 and a p-value of 0.0078. The reported Cohen\'s d was 2.92. The 95% confidence intervals also separated the two arms, with T2 ranging from 0.556 to 0.743 and T0 ranging from 0.318 to 0.447.

Looking beyond the composite score, T2 also stood out in the symptom breakdown. It had the lowest reversal rigidity score, 0.077, and the lowest drift-adaptation symptom, 0.010. Its impulsivity score, 0.193, was much lower than that of T0, which was 0.474. Credit assignment remained acceptable at 0.738, and jumpy learning stayed moderate at 0.734. In practical terms, this pattern suggests that the combination of entropy regularization and temperature annealing gave the policy enough freedom to adjust when the reward structure changed, but not so much freedom that training became erratic. It also helped the agent avoid the overspending failures that hurt several of the other arms.

The next best arm was T1, the simple baseline condition. Its health score was 0.573 and its mean return was 36.9. That is clearly better than untreated REINFORCE, but still below T2 on both main outcomes.

6.2 Performance of new RLHF arms: REINFORCE++ (T8) is the only scalable upgrade worth keeping

Among the three newer critic-free arms, T8, labeled REINFORCE++, performed best. Its mean health score was 0.548 with a standard deviation of 0.111, and its mean return was 32.4. Compared with T0, the gain in health was 0.166, with a p-value of 0.0156 and a reported Cohen\'s d of 1.50. This made it a clear improvement over untreated REINFORCE. At the same time, the difference between T8 and T2 was not significant after correcting for multiple comparisons. Even so, T8 remained competitive despite operating at a substantially higher dose, 95 compared with 35 for T2. Its drift-adaptation score, 0.093, was strong, and impulsivity was moderate at 0.307. The global running normalization used in this arm therefore seems to have reduced variance in a way that still allowed the policy to adjust after the preference shift, without requiring a learned critic.

T6, the RLOO arm, reached a health score of 0.493 and a mean return of 26.4. That was better than T0, but the improvement did not remain significant after Bonferroni correction, with a p-value of 0.0547. T7, the GRPO arm, performed much worse. Its health score fell to 0.207, and its mean return dropped to -256.5. Within this framework, that meant group-relative normalization by itself was not enough to maintain useful adaptation. Of the three newer RLHF-inspired variants, only T8 offered a clear improvement that looked both meaningful and durable.

6.3 Catastrophic failure of critic/PPO/GRPO arms (negative returns, perfect credit assignment but total rigidity & drift-blindness)

The weakest results came from the more elaborate arms that used a learned critic or stronger clipping. T3, T4, T5, and T7 all fit this pattern to different degrees, but T5, the PPO Cocktail arm, was the clearest example. It had the lowest mean health score, 0.192 with a standard deviation of 0.007, and a negative mean return of -24.7. Relative to T0, its health score was significantly worse, with a change of -0.191, a p-value of 0.0078, and a Cohen\'s d of -4.85. T3 also collapsed badly, ending with a health score of 0.346 and a return of -1043.3. T7 was similarly poor, with a health score of 0.207 and a return of -256.5. T4 was not quite as severe, but still disappointing, with a health score of 0.427 and a mean return of -10.7. It remained far behind T2.

One of the more striking findings is that these poor-performing arms often looked good on one symptom alone. For example, T5 had a credit-assignment score of 0.949, which was among the strongest values reported. But that came alongside maximal reversal rigidity, with a rigidity score of 1.000, and a high drift-adaptation symptom, 0.412. In other words, these methods looked organized on paper in terms of assigning return signal, yet failed badly when they had to change behavior after the environment shifted. The combination of critic-based estimates, clipping, and heavy variance control appears to have anchored the policy to the earlier, expensive actions and made it difficult to discover the newly better cheap actions once the reversal and drift occurred.

6.4 Symptom-level dissection (why entropy succeeds where variance-reduction overkill fails)

The symptom breakdown helps explain why T2 did so well. Its impulsivity score was the lowest in the full comparison, at 0.193. It also had the lowest reversal rigidity, 0.077, and the lowest drift symptom, 0.010. Credit assignment, at 0.738, was not the very best score in the table, but it was strong enough. This combination matters. T2 did not maximize one narrow metric at the expense of the others. Instead, it kept exploration active enough to remain adaptable while avoiding the complete instability associated with untreated learning.

The critic-based and clipped arms showed the opposite pattern. T3 through T5, along with T7, often produced the strongest credit-assignment values while showing the worst rigidity and drift scores. That suggests the problem was not an inability to propagate signal in a narrow statistical sense. The problem was that the policy became too tied to the structure it learned early in training. Once the environment changed, the stale critic or clipped objective no longer supported the kind of movement needed to find the new optimum. T8 sat between these extremes. It lowered impulsivity to 0.307 and kept drift reasonably low at 0.093, without showing the full collapse seen in the critic-heavy arms. The symptom-level picture therefore supports a simple conclusion: in this setting, moderate exploration helped preserve adaptation better than stronger forms of variance reduction.

6.5 Health-vs-dose curve: clear inverted-U confirming more intervention can induce rigidity

When the health scores are plotted against normalized dose, the resulting shape is an inverted U. Performance improves quickly from T0, which has a dose of 0, up to T2, which has a dose of 0.241. Around that point the curve levels off, with T1 and T8 still performing well, and then declines as the dose moves higher. Once the dose rises past about 0.5, health drops sharply. T5, which sits at the maximum dose of 1.000, ends with both the lowest health and one of the worst return profiles.

This pattern suggests that extra intervention does not produce a steady gain. Up to a point, added components help. Beyond that point, the same additions appear to make the policy more rigid and less able to adjust. The fact that this pattern holds across seeds suggests it is not just the result of a single hyperparameter choice or an unlucky run. Instead, it looks like a recurring response of the heavier methods to this particular combination of noise, reversal, and drift.

6.6 Learning-curve stability and late-episode adaptation analysis

The late-training return distributions, taken from the final 150 episodes, reinforce the same interpretation. T2 and T1 maintained positive returns and showed relatively stable behavior in the last part of training. By contrast, T3, T5, and T7 continued to produce negative returns and remained more variable. The narrow confidence interval around T5\'s health score is especially revealing. It means the poor outcome for PPO was not a rare collapse driven by one or two seeds. It was a repeated and consistent failure mode across runs.

T8 again occupied the middle position. Its late-stage behavior was more stable than the critic and PPO-based arms, and its returns recovered better after the shift, though not as cleanly as in T2. The general pattern is therefore clear. The methods that could continue adapting after the reward structure changed either improved or at least held their gains late in training. The heavier methods did not. They remained tied to policies that no longer fit the task.

Overall, the results point to T2 as the best-balanced treatment in this framework. It combined the highest health score, the highest return, and the strongest adaptation after reversal and drift. T8 was the only higher-dose alternative that remained broadly competitive. The critic-based and clipping-heavy arms, by contrast, underperformed consistently. In this environment, added complexity often brought less flexibility rather than more.

**Table 1. Health Score --- Paired Wilcoxon Signed-Rank Tests vs Untreated (T0)**

| **Treatment**   | **ΔHealth** | **p-value** | **Sig** | **Cohen's d** |
|-----------------|-------------|-------------|---------|---------------|
| T2 Entropy      | +0.267      | 0.0078      | \*\*    | 2.92          |
| T1 Baseline     | +0.190      | 0.0078      | \*\*    | 1.92          |
| T8 REINFORCE++  | +0.166      | 0.0156      | \*      | 1.50          |
| T6 RLOO         | +0.110      | 0.0547      | ns      | 1.08          |
| T4 Actor-Critic | +0.044      | 0.1953      | ns      | 0.42          |
| T3 Combined     | --0.037     | 0.3125      | ns      | --0.45        |
| T7 GRPO         | --0.175     | 0.0078      | \*\*    | --3.12        |
| T5 PPO          | --0.191     | 0.0078      | \*\*    | --4.85        |

*Note: Bonferroni α = 0.05/16 ≈ 0.0031; \*\*p \< 0.0031, \*p \< 0.05, ns = not significant. Negative Cohen\'s d indicates worse performance than T0.*

**Table 2. Health Score --- Paired Wilcoxon Signed-Rank Tests vs Best Arm (T2 Entropy)**

| **Treatment**   | **ΔHealth** | **p-value** | **Sig** |
|-----------------|-------------|-------------|---------|
| T1 Baseline     | --0.077     | 0.2500      | ns      |
| T8 REINFORCE++  | --0.101     | 0.0547      | ns      |
| T6 RLOO         | --0.156     | 0.1094      | ns      |
| T0 Untreated    | --0.267     | 0.0078      | \*\*    |
| T4 Actor-Critic | --0.223     | 0.0078      | \*\*    |
| T3 Combined     | --0.304     | 0.0078      | \*\*    |
| T7 GRPO         | --0.442     | 0.0078      | \*\*    |
| T5 PPO          | --0.457     | 0.0078      | \*\*    |

**Table 3. 95% Confidence Intervals (Health, t-distribution)**

| Treatment      | Mean Health | 95% CI Lower | 95% CI Upper |
|----------------|-------------|--------------|--------------|
| T2 Entropy     | 0.650       | 0.556        | 0.743        |
| T1 Baseline    | 0.573       | 0.512        | 0.633        |
| T8 REINFORCE++ | 0.548       | 0.449        | 0.648        |
| T0 Untreated   | 0.383       | 0.318        | 0.447        |
| T5 PPO         | 0.192       | 0.186        | 0.198        |

**Table 4. Overall Performance Metrics by Treatment Arm**

| Treatment          | Health (Mean ± SD) | Imp   | Crd   | Jmp   | Rig   | Dft   | Return  | Dose | NrmD  |
|--------------------|--------------------|-------|-------|-------|-------|-------|---------|------|-------|
| T0: Untreated      | +0.383 ± 0.072     | 0.474 | 0.819 | 1.000 | 0.686 | 0.106 | 23.3    | 0    | 0.000 |
| T1: Baseline (MPH) | +0.573 ± 0.067     | 0.408 | 0.800 | 0.702 | 0.170 | 0.057 | 36.9    | 27   | 0.186 |
| T2: Entropy (AMP)  | +0.650 ± 0.104     | 0.193 | 0.738 | 0.734 | 0.077 | 0.010 | 37.8    | 35   | 0.241 |
| T3: Combined Stim  | +0.346 ± 0.061     | 0.608 | 0.664 | 0.494 | 1.000 | 0.505 | -1043.3 | 106  | 0.731 |
| T4: Actor-Critic   | +0.427 ± 0.079     | 0.341 | 0.750 | 0.699 | 1.000 | 0.076 | -10.7   | 125  | 0.862 |
| T5: PPO Cocktail   | +0.192 ± 0.007     | 0.678 | 0.949 | 1.000 | 1.000 | 0.412 | -24.7   | 145  | 1.000 |
| T6: RLOO           | +0.493 ± 0.114     | 0.313 | 0.661 | 0.694 | 0.760 | 0.106 | 26.4    | 82   | 0.566 |
| T7: GRPO           | +0.207 ± 0.015     | 0.676 | 0.852 | 1.000 | 1.000 | 0.435 | -256.5  | 114  | 0.786 |
| T8: REINFORCE++    | +0.548 ± 0.111     | 0.307 | 0.672 | 0.635 | 0.551 | 0.093 | 32.4    | 95   | 0.655 |

*Note: Metrics aggregated across 8 seeds × 600 episodes. Preference drift = True.*

## 7. Discussion & Ablation Insights {#discussion-ablation-insights}

The results point to a fairly clear pattern. In this environment, which combines noisy rewards, uncertain reversal timing, and gradual preference change, the methods that used lighter interventions adapted better than those that added several layers of control. T2, the entropy-based arm, performed best not because it dominated every individual symptom by an extreme margin, but because it avoided the worst trade-offs. T8, the global-normalization variant, offered the strongest alternative among the higher-dose options. By contrast, the arms that relied on critics or stronger clipping often accumulated negative returns and showed the highest rigidity. This does not mean that critics or PPO-style methods are inherently poor choices in every setting. It does suggest, though, that in the type of non-stationary problem studied here, extra stabilization can work against adaptation rather than support it.

7.1 Mechanistic explanation: why critics and clipping create drift-blind policies

One likely reason is the way a learned critic responds to changing rewards. Early in training, before the reversal, expensive actions produce the highest rewards. A critic trained on that part of experience will naturally learn to favor those actions. Once the budget contracts and the reward ranking begins to drift, the critic does not immediately reflect the new situation. If the policy update continues to depend on that lagging estimate, then the gradient signal begins to point backward, toward a pattern that was once useful but no longer is. In that case, the policy may look well organized in terms of conventional credit-assignment statistics while still adapting badly to the changed environment. That is close to what happened here. The critic-based arms often had strong credit-assignment scores but still showed extreme rigidity and weak drift adaptation.

Clipping seems to intensify the same problem. PPO and related approaches are designed to prevent large, potentially harmful policy shifts, which is often valuable in stable tasks \[6\]. But the environment in this study does not remain stable. After reversal, the policy sometimes needs to move decisively enough to sample the now-better cheap actions. Clipping reduces that freedom. Even with entropy present, the policy remains pulled toward the earlier action pattern. Batching and lower-noise settings can reinforce the same tendency by dampening the randomness that might otherwise help the agent discover the new reward structure. The contrast with T2 and T8 is useful here. Those arms worked without a stale critic and without the same hard constraints on policy movement, which seems to have made them more responsive once the reward landscape changed. The symptom pattern fits that reading: the critic and clipping arms often posted rigidity scores of 1.000 and drift symptoms above 0.40, despite otherwise strong-looking internal statistics.

7.2 Clinical/RL analogy: gentle stimulants vs heavy antipsychotics

A rough analogy helps make the pattern easier to picture. The entropy term combined with temperature annealing behaves somewhat like a mild stimulant. It reduces the worst forms of impulsive randomness without shutting down flexibility. In the results, impulsivity dropped sharply from 0.474 in T0 to 0.193 in T2, yet T2 still retained the ability to respond after reversal. The system became calmer, but not rigid.

The heavier arms behaved more like a strong sedative response. On paper, some surface indicators improved. Variance could appear more controlled, and credit assignment could even look better. But the policy became less responsive to what was happening around it. Once the environment changed, it kept following the old pattern. That is why several of the critic-based or clipping-based arms looked tidy in one part of the table but still ended up with negative returns and poor adaptation. The analogy is imperfect, of course, because reinforcement-learning algorithms are not clinical subjects. Even so, it captures the main point: too much stabilization can suppress the very flexibility the task requires.

7.3 Implications for real-world RLHF (prompt drift, human preference shifts, robotics)

The broader relevance of this pattern is not limited to the toy environment. In RLHF, preference targets do not stay perfectly fixed. Annotation instructions may change, norms may shift, and the kinds of prompts seen in deployment can evolve over time \[7\]. Ahmadian et al. \[2\] argue from a different angle that PPO has become the default in RLHF despite high computational cost and sensitive tuning, and they show that simpler REINFORCE-style methods can outperform PPO as well as newer \"RL-free\" alternatives. The present results fit well with that broader claim. If the environment or the target preference is changing, methods that are too tightly controlled may become slow to adapt at exactly the wrong moment.

The same lesson carries over to other domains. In robotics, wear, payload changes, or new obstacles can shift what counts as a good action. In interactive systems, user behavior can drift gradually rather than switch cleanly. In such cases, starting with a lighter method may be safer than immediately deploying a full actor-critic or PPO stack. A simple entropy-based setup can preserve responsiveness at lower cost, and a critic-free normalization method such as the one used in T8 may be worth adding if drift appears to be the main problem. That line of reasoning is close to the \"start simple\" message that has recently re-emerged in RLHF work \[2\].

7.4 Limitations (simplified MDP, fixed drift schedule) and sensitivity analysis notes

Several limits of the study should be kept in view. The environment is intentionally small, discrete, and fully observable. That makes the failure modes easier to isolate, but it is also much simpler than most real tasks. Larger problems may involve continuous actions, hidden state, and much longer horizons. The same ranking of treatment arms may not carry over unchanged to those settings.

The drift schedule is also simple. It is linear and tied directly to the reversal point. In practice, preference change may be slower, noisier, or driven by outside events rather than by an internal schedule. Another limit is that the study did not include broad hyperparameter sweeps within each arm. The comparison relies on a fixed set of defaults carried over from earlier stages of the pipeline. Some sensitivity checks were run, and they suggested that moderate changes in entropy decay or clipping thresholds changed the absolute numbers without overturning the overall ranking. Still, the conclusions should be taken as informative rather than universal. The health metric has its own simplifying choice as well: it weights all five symptoms equally. A different weighting scheme could shift the ranking at the margins.

7.5 Practical prescription: start with T2-style entropy; escalate only to REINFORCE++ when proven necessary

From a practical standpoint, the dose-response pattern suggests a simple starting rule. When a new task is likely to involve noisy gradients and some degree of non-stationarity, begin with an entropy-regularized REINFORCE setup of the T2 kind. Track signs of rigidity and weak drift adaptation during early runs. If those remain a problem after ordinary tuning, the next step should be something like the T8 mechanism, which adds global running normalization without bringing in a learned critic.

The results do not support treating critic-based or clipped methods as the default first choice in this kind of setting. Those tools may still help in some domains, but the evidence here suggests they should be introduced only after ablations show that they improve not just internal stability, but actual return and adaptation. A staged approach of this kind keeps the compute burden lower and reduces the chance of locking the policy into an outdated pattern.

## 8. Conclusion & Future Work {#conclusion-future-work}

The main result of the study is straightforward. In a setting that combines reward noise, random reversal, and gradual preference drift, the best-performing treatment was not the most elaborate one. It was the simpler entropy-based arm, which achieved the highest health score, the strongest adaptation profile, and the best mean return. Adding more machinery often reduced flexibility enough to outweigh any stabilization benefit. Among the newer critic-free methods, only the global running normalization variant remained clearly competitive without collapsing.

These findings do not overturn the value of actor-critic or PPO-style methods in stable environments. They do, however, reinforce the case for caution when conditions are changing. Williams \[1\] introduced REINFORCE as a general gradient-following family for stochastic reinforcement learning, and recent RLHF work has argued that simpler REINFORCE-style methods can still be highly effective in modern alignment settings \[2\]. The present results support that broader lesson. Simplicity is not just cheaper. Under some forms of non-stationarity, it may also be more robust.

Future work will extend the pipeline in several directions. One next step is to test hybrid arms that combine light entropy regularization with global running normalization, since that may preserve the main strengths of T2 and T8 at the same time. Other planned extensions include continuous-control settings, alignment-style simulations with more realistic prompt drift, and multi-agent tasks where preference changes do not occur at the same time for every agent. By releasing the codebase and JSON logs, the study also aims to make these comparisons easier to reproduce and extend in other domains.

References

\[1\] Williams RJ. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Mach Learn*. 1992;8(3-4):229-256. doi:10.1007/BF00992696

\[2\] Ahmadian A, Cremer C, Gallé M, et al. Back to basics: revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In: Ku LW, Martins A, Srikumar V, eds. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics; 2024:12248-12267. doi:10.18653/v1/2024.acl-long.662

\[3\] Sutton RS, Barto AG. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press; 2018.

\[4\] Mnih V, Badia AP, Mirza M, et al. Asynchronous methods for deep reinforcement learning. In: *Proceedings of the 33rd International Conference on Machine Learning*. PMLR; 2016:1928-1937.

\[5\] Haarnoja T, Zhou A, Abbeel P, Levine S. Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor. In: *Proceedings of the 35th International Conference on Machine Learning*. PMLR; 2018:1861-1870.

\[6\] Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. Preprint. 2017. doi:10.48550/arXiv.1707.06347

\[7\] Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*. 2022;35:27730-27744.

\[8\] Shao Z, Wang P, Zhu Q, et al. DeepSeekMath: pushing the limits of mathematical reasoning in open language models. Preprint. 2024. doi:10.48550/arXiv.2402.03300

\[9\] Cheung WC, Simchi-Levi D, Zhu R. Hedging the drift: learning to optimize under nonstationarity. *Management Science*. 2022;68(10):7197-7216. doi:10.1287/mnsc.2021.4024

\[10\] Khetarpal K, Riemer M, Rish I, Precup D. Towards continual reinforcement learning: a review and perspectives. Preprint. 2020. doi:10.48550/arXiv.2012.13490

\[11\] Wang T, Xia Z, Chen X, Liu S. Tracking drift: variation-aware entropy scheduling for non-stationary reinforcement learning. Preprint. 2026. arXiv:2601.19624.
