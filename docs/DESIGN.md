# Survival Sim: Design

## Goal
Observe how reinforcement learning (RL) agents learn and evolve in a primitive, Earth-like world. Agents are told nothing about resources, dangers, hunting or farming. They must discover everything through experience.

## Agent Instincts (Rewards)
Agents only have instincts, given to them as rewards. In order of importance:

| Instinct | Reward | Strength |
|---|---|---|
| Survive | Small reward for every tick alive, large penalty for dying | Primary |
| Breed | Large reward when a child is born | Primary |
| Comfort | Small bonus when fed, watered, warm, rested and safe | Secondary (much smaller) |

Nothing else is rewarded. Eating, hunting, farming, crafting, fighting and avoiding danger are never rewarded directly; agents must learn whether these help them survive and breed. Violence between agents is neither rewarded nor punished, so whether agents become peaceful, cooperative or violent is up to them.

## World
- Grid: 128 x 128 tiles
- Terrain: grass, forest (berry bushes), rock, river, pond, deep water, cliff, cave, farmland
- Agents see only a few tiles around them (limited vision)
- Viewer has a movable, zoomable camera

## Time
- 1 tick = one action by every agent
- 1 day = 100 ticks (70 day, 30 night)
- 1 year = 20 days, with 4 seasons (food scarce in winter)
- Training runs as fast as possible with no graphics
- Viewing mode runs at a chosen speed

## Population
- Starting agents: 200 (100 male, 100 female)
- Population cap: 1,000
- Lifespan: about 3 to 5 years
- Adult (able to breed) from age 1 year
- Children are slower and weaker until adulthood

## Agents
- Stats: health, hunger, thirst, energy, warmth, age
- Sex: male or female
- Inventory: can carry a small amount (meat, berries, seeds, sticks, stones, one weapon)
- Actions: move (4 directions), rest, eat, drink, pick up, plant, craft, attack animal, attack person, mate
- The attack action works on animals and on other agents alike; agents are not told who or what they should attack

## Food Sources (all must be discovered)
| Source | How it works | Trade-off |
|---|---|---|
| Foraging | Eat berries from forest tiles | Easy, but limited and none in winter |
| Hunting | Attack animals to get meat | More food, but prey runs away and costs energy |
| Farming | Eating berries gives seeds; planting seeds on grass near water grows crops over several days | Slow, delayed reward, but the most food in the long run |

Note: farming is the hardest to discover because the reward comes days later. There is no time limit; however long it takes agents to discover it (if ever) is part of the result.

## Weapons (must be discovered)
Agents are not told that weapons exist. They must discover that picking things up and using the craft action makes them stronger.

| Weapon | How to get it | Effect |
|---|---|---|
| Bare hands | Default | Weak damage |
| Stick | Pick up in a forest | Slightly more damage |
| Sharp stone | Craft while holding a stone next to rock | Medium damage |
| Spear | Craft while holding a stick and a sharp stone | High damage, can hit from 2 tiles away |

- Weapons wear out after a number of uses and break
- Weapons make hunting wolves and bears more realistic, but still hard
- A dead agent drops its inventory, so others can take its food and weapon

## Animals
Animals follow simple fixed rules (they do not learn).

| Animal | Type | Behaviour |
|---|---|---|
| Rabbit | Prey | Fast, small amount of meat, breeds quickly |
| Deer | Prey | Flees when agents come near, more meat |
| Wolf | Predator | Hunts in packs, attacks prey and agents, active mostly at night |
| Bear | Predator | Rare, very strong, guards forest areas |

Wolves and bears can also be hunted, but it is very hard: they have high health, fight back hard, and a lone agent will usually lose. Several agents attacking together have a better chance. A bear gives the most meat of any animal.

Prey eat grass and breed; predators eat prey (and agents) and breed. Their populations rise and fall naturally, forming a food chain.

## Dangers (all active from the start)
| Danger | Effect |
|---|---|
| Starvation and thirst | Health drains when hunger or thirst is empty |
| Cold | Night and winter drain warmth unless in shelter or near fire |
| Predators | Wolves and bears attack and damage agents |
| Hunting injuries | Animals fight back; wolves and bears can easily kill a hunter |
| Poisonous plants | Look like food, cause damage |
| Dirty water | Ponds may cause sickness |
| Deep water and cliffs | Instant death |
| Disasters | Random storms, floods and wildfires |
| Exhaustion | Agent cannot move at zero energy until it rests |
| Other agents | Compete for food; can attack and kill each other, with or without weapons |
| Old age | Agents die naturally at the end of their lifespan |

If the population keeps going extinct, numbers are tuned (for example fewer predators or more berries). Dangers are never removed.

## Learning
Two layers, like real humans:

1. **Shared learning (culture):** all agents share one neural network "brain". Every agent's experience, including mistakes and deaths, trains it, so agents learn from each other.
2. **Evolution (genes):** each agent has genes: small personal adjustments to the shared brain plus traits (speed, vision range, strength, cold resistance). A child gets a mix of both parents' genes with small random mutations. Agents that survive and breed more pass on their genes more.

## Breeding
- Two adults (age 1 year or more) of opposite sex, next to each other, both choose "mate" within a few ticks of each other
- Both must have food, water and health above 50
- Pregnancy lasts 3 days; the mother uses 50% more food and energy during it
- After giving birth, a mother cannot conceive again for 5 days; there is a small (2%) risk of dying in childbirth
- A child starts on the mother's tile, moves at half speed and hits at 30% strength until age 1 year
- Population cap: 1,000 alive at once

## Genes
Every agent is born with four genes. Higher is better, but each point above normal costs extra food (a bigger, tougher body needs more fuel), so evolution has to find a balance.

| Gene | Effect |
|---|---|
| Strength | Damage dealt in fights and hunts |
| Hardiness | Resistance to cold |
| Endurance | Less energy used when walking |
| Vision | How far the agent can see (used by the brain in phase 10) |

A child gets the average of both parents' genes plus a small random mutation.

## Measurements (what we record)
- Population over time, births and deaths per day
- Causes of death
- Average lifespan per generation
- How often agents forage, hunt, farm, shelter, craft
- Weapon use, and agents killed by other agents
- Animal populations over time
- Gene trait averages over generations

## Build Phases
Each danger and feature is coded and tested one at a time, but experiments run with everything switched on.

1. Project setup (done)
2. World grid with food, water, day/night and seasons (done)
3. Pixel-art viewer (done)
4. Bigger 128 x 128 world with all terrain types, and a camera for the viewer
5. Agents: stats, movement, eating, drinking, death (random actions, no learning yet)
6. Animals: prey and predators with simple rules
7. Remaining dangers: cold, poison, dirty water, cliffs, disasters, exhaustion
8. Hunting, farming, crafting and weapons, and agents attacking each other
9. Breeding, ageing and genes
10. Learning: shared brain trained with RL
11. Evolution across generations
12. Recording measurements and charts
13. Running the experiment and tuning
