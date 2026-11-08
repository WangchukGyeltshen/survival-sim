# Survival Sim: Design

## Goal
Observe how reinforcement learning (RL) agents learn, cooperate, fight and evolve in a primitive, Earth-like world. Agents are told nothing about resources, dangers, hunting, farming, tools or building. They must discover everything through experience.

## Long-term plan (curriculum)
1. **Train** the shared brains in a 128 x 128 world.
2. **Make the world more realistic step by step** while training continues (each upgrade carries the brain's knowledge over with "brain surgery").
3. **Final experiment:** after training is complete, one long recorded history run in a much bigger world with continents, oceans and biomes.

Two rules keep this possible:
- The brain only ever sees its **local surroundings** (a 15 x 15 view), never the whole map, so a brain trained on a small world works on a big one.
- New senses (view channels, self numbers) and new actions are **only ever added at the end**, never reordered or removed, so every upgrade can reuse the old brain.

## Agent Instincts (Rewards)
Agents only have instincts, given to them as rewards:

| Instinct | When | Reward | Strength |
|---|---|---|---|
| Survive | every tick alive | +0.01 (+1 per day) | Primary |
| Survive | dying (except of old age) | -3 | Primary |
| Breed | a child is born (both parents) | +5 | Primary |
| Parenting | each tick, per own child under 1 year that is alive | +0.002 (+0.2 per day) | Secondary |
| Comfort | fed, watered, warm, rested, healthy, not sick, no pain | +0.002 (+0.2 per day) | Secondary |

Nothing else is rewarded. Eating, hunting, farming, crafting, building, sharing, fighting and avoiding danger are never rewarded directly; agents must learn whether these help them survive and breed. Violence is neither rewarded nor punished.

## World
- Grid: 128 x 128 tiles (bigger worlds with continents planned for the final run)
- Terrain: grass, forest (berry bushes), rock, river, pond, deep water, cliff, cave, farmland
- Resource deposits: flint, clay, copper ore, tin ore, iron ore
- Structures built by agents (see Building)
- Agents see only a few tiles around them (vision gene, less at night)

## Time
- 1 tick = one action by every agent
- 1 day = 100 ticks (70 day, 30 night)
- 1 year = 20 days, with 4 seasons (food scarce in winter, no crop growth)

## Population and tribes
- Starting agents: 200, as **4 tribes of 50**, one tribe in each quarter of the map
- Each tribe has its own clothing colour; children join their mother's tribe
- Agents can see whether a person belongs to their own tribe or another
- Population cap: 1,000; lifespan about 3 to 5 years; adult from age 1 year
- Children are slower and weaker until adulthood

## Agents
- Stats: health, food, water, energy, warmth, pain, sickness, age
- Sex: male or female; genes: strength, hardiness, endurance, vision
- Inventory: berries, meat, cooked meat, seeds, sticks, stones, flint, clay, copper ore, tin ore, iron ore, copper, bronze, iron, a pot (holds 3 drinks of water), one weapon, and tools (pick, sickle)

### Actions (19)
| # | Action | What it does |
|---|---|---|
| 0 | rest | regain energy |
| 1-4 | up, down, left, right | move (also sets the direction the agent faces) |
| 5 | eat | from the tile (berries, crop) or carried food (cooked meat, meat, berries) |
| 6 | drink | from water or a well next to you, or from a carried pot |
| 7 | pick up | piles, storage, crops, berries, flint, clay, sticks, stones, ores |
| 8 | plant | plant a seed on grass |
| 9 | craft | make the best thing your items and surroundings allow (see Technology) |
| 10 | attack animal | hunt; wolves and bears bite back |
| 11 | attack person | hurt another person; with nobody in reach it damages a structure instead |
| 12 | mate | try to conceive with a willing partner next to you |
| 13 | give | hand food (or a material) to the hungriest person next to you; on a storage tile with nobody near, store your goods |
| 14 | build | build your current blueprint on the tile you face (or repair / refuel it) |
| 15 | next plan | switch your blueprint to the next structure type |
| 16-18 | call 1, call 2, call 3 | make a sound heard within 6 tiles; meanings are never defined |

## Food Sources (all must be discovered)
| Source | How it works | Trade-off |
|---|---|---|
| Foraging | Eat berries from forest tiles | Easy, but limited and none in winter |
| Hunting | Attack animals to get meat | More food, but risky; raw meat can make you sick |
| Cooking | Craft raw meat next to a campfire | Cooked meat gives more food and is safe |
| Farming | Eating berries gives seeds; planted seeds grow into crops (faster near water) | Slow, delayed reward; a sickle doubles the harvest |
| Storing | Put food in storage | Food in storage keeps much longer, for winter |

## Technology (each step must be discovered, in order)
```
STONE AGE    stick (forest) -> sharp stone (stone + rock) -> spear (sharp stone + stick)
             flint -> flint blade
FIRE         campfire = build with 1 flint + 2 sticks
POTTERY      clay next to a fire -> pot (carry 3 drinks of water)
COPPER AGE   copper ore next to a fire -> copper; copper + stick -> copper spear
BRONZE AGE   copper + tin ore next to a furnace -> bronze -> bronze spear, bronze pick, bronze sickle
IRON AGE     iron ore (needs a pick to mine) next to a furnace -> iron -> iron spear, iron pick, iron sickle
```

| Weapon | Damage | Uses before breaking |
|---|---|---|
| hands | 5 | - |
| stick | 9 | 20 |
| sharp stone | 15 | 30 |
| spear (reach 2) | 25 | 40 |
| flint blade | 18 | 40 |
| copper spear (reach 2) | 32 | 60 |
| bronze spear (reach 2) | 40 | 90 |
| iron spear (reach 2) | 50 | 140 |

Tools: a **pick** (bronze or iron) is needed to mine iron ore; a **sickle** doubles crop harvests.

## Building
Each agent holds a **blueprint** (a plan in their head). "next plan" switches it; "build" builds it on the tile the agent faces, if they carry the materials. Building on an existing structure of the same kind repairs it (campfires: adds fuel).

| Structure | Materials | Effect | Lasts |
|---|---|---|---|
| Campfire | 1 flint + 2 sticks | warmth within 2 tiles, wolves avoid it at night, cooking, smelting copper | 3 days (add sticks) |
| Hut | 4 sticks | warm shelter, better rest, wolves and bears can't enter | 30 days |
| Wooden fence | 2 sticks | blocks animals and people | 20 days |
| Stone wall | 2 stones | blocks animals and people | 100 days |
| Storage | 2 sticks + 1 clay | holds items; stored food keeps much longer; anyone can take from it | 30 days |
| Bridge | 3 sticks, on water | walk across water safely | 40 days |
| Well | 3 stones + 1 clay | clean water source anywhere | 100 days |
| Watchtower | 4 sticks + 2 stones | standing on it gives +3 vision | 40 days |
| Furnace | 3 stones + 2 clay, next to a fire | smelting bronze and iron | 100 days |

Structures wear out (wood faster than stone). Attacking with nobody in reach damages an adjacent structure, so walls can be broken and storage raided. Campfires near forests can occasionally start wildfires.

## Animals
Animals follow simple fixed rules (they do not learn).

| Animal | Type | Behaviour |
|---|---|---|
| Rabbit | Prey | Fast, small amount of meat, breeds quickly |
| Deer | Prey | Flees, more meat; herds **migrate with the seasons** (north in summer, south in winter) |
| Wolf | Predator | Hunts in packs, mostly at night; avoids campfires |
| Bear | Predator | Rare, very strong, guards forest areas |

Wolves and bears can be hunted, but it is very hard; groups and better weapons help. Huts keep predators out.

## Dangers (all active from the start)
| Danger | Effect |
|---|---|
| Starvation and thirst | Health drains when food or water is empty |
| Cold | Night, winter and storms drain warmth unless in a cave, hut, forest, huddled or near fire |
| Predators | Wolves and bears attack |
| Hunting injuries | Animals fight back |
| Poisonous plants | Purple berries hurt a lot |
| Dirty water | Pond water may cause sickness |
| Disease | Sickness spreads to people standing next to a sick person; random outbreaks; raw meat |
| Deep water and cliffs | Instant death (bridges make water safe) |
| Disasters | Storms, lightning, floods, wildfires |
| Exhaustion | No movement at zero energy |
| Other agents | Can attack, kill and loot; can break structures |
| Old age | Natural end of life |

## Breeding and genes
- Two adults of opposite sex, next to each other, both choose "mate" within a few ticks; both must be fed, watered and healthy
- Pregnancy 3 days; the mother rests 5 days after birth; 2% risk of dying in childbirth
- Body genes: strength, hardiness, endurance, vision. Higher is better but costs extra food. A child gets the average of both parents plus a small mutation.
- Behaviour genes (Stage 3), an inborn temperament from -1 to +1 (founders start near 0). They cost nothing; they nudge the brain's action scores before choosing (`src/temperament.py`). A gene of +1 makes its actions about 2.7 times as likely; the brain can still overrule it.

| Gene | Nudges |
|---|---|
| aggression | attack a person (and a little: attack an animal) |
| generosity | give |
| drive | mate |
| industry | craft, build, plant |
| curiosity | all choices more random (higher) or more fixed (lower) |

## What agents sense
- **View:** a 15 x 15 square of tiles, with one channel per question: ground type, berries (red / purple), crops, fire, flood, ash, piles, animals, people (man, woman, child, armed), visibility, resource deposits, each structure type, sick person, own-tribe person, other-tribe person, and calls heard
- **Self:** body stats, pain, sickness, pregnancy, age, sex, everything carried, weapon and tools, body genes, time of day, season, storm, last action, blueprint, direction faced, calls heard, tribe, behaviour genes (99 numbers)

## Learning
1. **Tribe learning (culture):** agents learn with PPO (an actor-critic method). Each tribe has **its own brain** (Stage 3), trained only on its own members' experience with its own optimizer, so tribes can develop different habits. Children join their mother's tribe and use its brain. Resuming a run saved with one shared brain splits it: all 4 tribe brains start as copies and then grow apart. `train.py --brains 1` starts a run with one brain shared by everyone (for comparison).
2. **Evolution (genes):** inherited genes shape bodies and, through behaviour genes, temperament. Evolution favours whatever temperament survives and breeds best; the shared brain learns how to make the most of each one.

3. **Memory (Stage 3):** a GRU layer keeps 128 numbers per agent from one tick to the next, so an agent can remember what it can no longer see. A newborn starts with an empty memory. While learning, decisions are replayed in 16-tick stretches of one agent's life, so the brain learns what was worth remembering.
4. **Brain surgery (Stage 3):** when the world gains senses or actions, `surgery.py` grows the old brain into the new shape. Old weights are copied; new senses start ignored (zero weights); new actions start with a small share of the chances (10% in total); the memory starts ignored. A check confirms the grown brain chooses exactly like the old one. run2 started from run1 this way. A run can also be upgraded **in place** (same run, round count kept, old brain backed up), which is the normal step of the long-term plan.
5. **Parallel worlds (Stage 4):** training runs several worlds at once (4 by default, `--worlds N`). World 0 runs in the main process and is the one the window shows; the others run in worker processes on other CPU cores. Every tick, all agents in all worlds decide in one go on the GPU. Each learning round therefore mixes decisions from several worlds, seasons and situations, and collecting them takes little longer than for one world. Each world resets on its own (fewer than 20 alive, or after 100 days).
6. **Comparison experiments (Stage 4):** `compare.py` draws 2 or more runs on the same charts with a report page. Training runs are lined up by decisions trained (survival, reward, curiosity, births, speed, causes of death); history runs by day (population, births, deaths, generations, weapons, violence, food work, structures, sharing, genes, causes of death). Switches for one-change comparisons: `--brains 1` (one shared brain instead of tribe brains) and `--no-temperament` (behaviour genes inherited but not nudging choices); trained vs untrained brains in the same world use `experiment.py --seed`.

## Measurements
- Training: survival, reward, curiosity, births, causes of death (`analyze.py runs/<name>`)
- History runs: population by tribe, births and deaths, causes of death, what agents do, violence, hunting, crafting, farming, building, sharing, calls, body and behaviour genes over generations, animals, heatmap, family trees (`analyze.py experiments/<name>`)

## Build stages
1. World, viewer, agents, animals, dangers (done)
2. Hunting, farming, crafting, weapons, fighting, breeding, genes (done)
3. Learning: senses, rewards, brain, PPO trainer (done)
4. Stage 1 tools: history runs, replays, charts, family trees, heatmaps (done)
5. Stage 2: the bigger world: resources and metal ages, fire, building, disease, migration, sharing, calls, tribes, parenting (done)
6. Stage 3: the bigger brain: memory, brain surgery, behaviour genes, tribe brains (done)
7. Stage 4: faster training with several worlds at once, comparison experiments (done)
8. **Next:** keep adding realism while training continues
9. Final experiment: big world with continents
