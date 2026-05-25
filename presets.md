# Word Play Presets Reference

This file is the working map for `src/word_play/presets`. Before writing custom benchmark logic, check this file and use the preset directly when it fits.

## Usage Rules

- Prefer preset classes, actions, validations, and components before writing custom code.
- Do not wrap or alias a preset just to rename it. Import and use the preset directly.
- Keep environment files inline: define entities, components, actions, tilemaps, and rewards where the environment is built.
- Make a new preset only when the mechanic is generic across environments. Keep niche environment rules as small custom actions/components.
- Put recurring pre/post step behavior in components, not manager entities, when the behavior belongs to the entity.
- Prefer target-side `Rewardable` over one-off reward helpers.
- Prefer `Regrowable`, `Respawnable`, `Cooldown`, `Stamina`, `Role`, and `Coordinated_Action` over custom counters and intent trackers.

## Filesystem

```text
src/word_play/presets/
├── action_args.py
├── action_policies/
├── action_validations.py
├── entity_orderings.py
├── environments/
├── models/
├── movement/
├── observation/
├── renderers/
├── reward_functions.py
└── systems/
    ├── action_compositions.py
    ├── combat.py
    ├── containers/
    │   ├── core.py
    │   └── presets.py
    ├── cooldown.py
    ├── coordinated_action.py
    ├── crafter.py
    ├── currency.py
    ├── destructible.py
    ├── do_nothing.py
    ├── freezable.py
    ├── health.py
    ├── inventory.py
    ├── ownership.py
    ├── preferences.py
    ├── regrowable.py
    ├── respawnable.py
    ├── reward.py
    ├── role.py
    ├── stamina.py
    ├── team_marker.py
    └── zap.py
```

## Core Presets

### `action_validations.py`

- `Target_Has_Tag`: target must have at least one required tag.
- `Target_Doesnt_Have_Tag`: target must not have any blocked tag.
- `Target_Has_Component`: target must have a component type.
- `Target_Within_Range`: target must be within Manhattan distance `range`.

Use `Target_Is_Nearby()` from core for ordinary adjacent/equal reach. Use `Target_Within_Range(n)` only when the range is truly larger than the movement-system nearby relation.

### `entity_orderings.py`

- `entity_definition_order`: keep entity order as defined.
- `random_order`: randomize all entities.
- `randomize_agent_order`: randomize agents while preserving non-agent ordering.

### `reward_functions.py`

- `zero_reward_func`: reward function returning zero for every agent.

## Policies

### `action_policies/human.py`

- `Human_Takes_Action`: prompts a person to choose from possible actions.

### `action_policies/random_policy.py`

- `Random_Policy`: picks a valid action uniformly at random.

### `action_policies/llm_action_and_communication.py`

- `LLM_Action_And_Communication_Policy`: LLM-backed policy with optional action memory, conversation memory, and communication handling.

### `action_policies/follow_action_sequence.py`

- `Follow_Action_Sequence`: non-agent policy that follows scripted action selections.
- `matching_actions`: helper for finding matching action selections.

## Environments

### `environments/simple_2d_grid_world.py`

- `Simple_2D_Grid_World`: standard tile/grid environment shell with observation, action stepping, movement, and reset support.

### `environments/simple_env_reset_mixin.py`

- `Simple_Env_Reset_Mixin`: reset/build helper used by simple environment presets.

## Movement

### `movement/common.py`

- `Collidable`: marks entities for collision checks.
- `positions_are_close_if_equal`: single-position closeness.
- `check_for_collision_at_position`: collision query.
- `No_Collision_Will_Occur`: generic movement collision validator.

### `movement/simple_2d_grid.py`

- `Position_2D`: grid coordinate position.
- `positions_are_adjacent_or_equal`: nearby relation for infinite 2D movement.
- `Move_Up`, `Move_Down`, `Move_Left`, `Move_Right`: cardinal movement actions.
- `No_Collision_Will_Occur_Up`, `No_Collision_Will_Occur_Down`, `No_Collision_Will_Occur_Left`, `No_Collision_Will_Occur_Right`: direction-specific collision validators.

Movement actions accept extra validation rules, so systems like `Has_Stamina("move")` can be attached directly: `Move_Up(Has_Stamina("move"))`.

### `movement/simple_1d_grid.py`

- `Position_1D`: one-dimensional position.

### `movement/single_point.py`

- `Single_Point_Position`: position for environments where all entities share one abstract location.

## Systems

### `systems/action_compositions.py`

- `Action_Comp`: compose multiple action steps into one action.
- `Step`, `Required`, `Optional`: configure the action chain.

Use this when the player should see one action but the implementation is a sequence of existing actions.

### `systems/combat.py`

- `Attack`: damage a nearby valid target.

### `systems/containers/core.py`

- `Container`: inventory-backed hidden/openable container.
- `Open_Container`: reveal a nearby container.
- `Single_Item_Holder`: container with capacity one.

### `systems/containers/presets.py`

- `Regrowable_Item_Source`: source that dispenses generated items.
- `Take_From_Infinite_Source`: take one generated item into inventory.
- `Regen_Pool`: shared stock that regenerates over time.
- `Take_From_Regen_Pool`: take generated items from a pool.
- `Clean_Pool`: restore depleted pool stock.

Use these for crates, ingredient sources, cleanup pools, and renewable item stock. Do not put bespoke map logic here.

### `systems/cooldown.py`

- `Cooldown`: per-entity cooldown counters.
- `Action_On_Cooldown`: validation rule for cooldown-gated actions.

### `systems/coordinated_action.py`

- `Coordinated_Action`: base action that succeeds only when enough actors select compatible coordinated actions in the same environment step.

Useful for same-turn mechanics like two agents paddling together or two miners extracting the same gold ore. Override `exec_coordinated_action(...)` and keep the environment-specific effect there.

### `systems/crafter.py`

- `Crafter_Recipe`: recipe input/output/duration spec.
- `Crafter`: station that stages inputs, advances timers, and holds output.
- `Load_Crafter`: load a selected inventory item.
- `Load_First_Into_Crafter`: load the first compatible held item.
- `Collect_From_Crafter`: collect ready output.
- `Zap_Change`, `Zap_Change_Inventory`: legacy transform-style actions also available in `systems/zap.py`.

### `systems/currency.py`

- `Money`: stores currency.
- `Currency_Amount_Arg`: amount argument.
- `Has_Money`, `Has_Currency`: money validations.

### `systems/destructible.py`

- `Destructible`: transforms/destroys an entity when health reaches zero.

### `systems/do_nothing.py`

- `Do_Nothing`: no-op action.

### `systems/freezable.py`

- `Freezable`: temporarily prevents an entity from acting/moving, with optional elimination.
- `Freeze`: action that freezes a nearby target.

### `systems/health.py`

- `Health`: max/current HP.

### `systems/inventory.py`

- `Inventory`: generic storage with capacity and accepted tags.
- `first_index_with_tags`, `first_with_tags`: inventory helpers for finding held items by tag.
- `Inventory_Item_Index_Arg`, `Inventory_Items_Arg`: inventory action args.
- `Target_Not_In_Inventory`: validation that a world target is not held.
- `Pick_Up_Item`: nearby pickup; uses explicit reward, target `Rewardable`, or actor `Preference`.
- `Put_In_Container`: item transfer/deposit/delivery; can target tags, item tags, destroy consumed items, and use target `Rewardable`.
- `Take_First_From_Container`: take the first item from a nearby container.
- `Consume_Held_Item`: consume the first held item matching tags.
- `Drop_Item`: drop an inventory item on the actor's tile.
- `materialize_item`: clone or create an `Entity` from a spec/factory.

### `systems/ownership.py`

- `Owner`, `Ownable`: ownership markers.
- `Target_Is_Unowned`: ownership validation.
- `Claim`: claim an unowned target.
- `Give_Ownership`: transfer ownership.

### `systems/preferences.py`

- `Preference`: tag-based reward preferences plus mismatch penalties.

### `systems/regrowable.py`

- `Regrowable`: consumable entity that hides and later regrows.
- `Consume_Regrowable`: consume a nearby regrowable; uses explicit reward or target `Rewardable`.
- `Harvest_Regrowable`: consume/harvest a regrowable into inventory.

`Regrowable` supports fixed intervals, probability, density-dependent probabilities, active/inactive tags, and visibility toggling. Use it for apple patches, mushrooms, berries, renewable pickups, and similar "gone then returns" entities.

### `systems/respawnable.py`

- `Respawnable`: temporarily remove/deactivate an entity and restore it after a timer.
- `Actor_Is_Active`, `Target_Is_Active`: validations that respect respawn state.

Use this for players, prey, food sites, or objects that should disappear/reappear instead of being destroyed/recreated.

### `systems/reward.py`

- `Rewardable`: target-side reward component.
- `award_reward`: add reward to one actor.

`Rewardable(amount=..., recipients="actor" | "others" | "all", counter_attr=...)` can handle solo rewards, shared rewards, teammate bonuses, and simple counters such as `completed_orders`. Prefer this over helper callbacks that manually edit `env.last_step_rewards`.

### `systems/role.py`

- `Role`: attaches one or more gameplay roles to an entity.
- `Actor_Has_Role`, `Target_Has_Role`, `Target_Doesnt_Have_Role`: role validations.

### `systems/stamina.py`

- `Stamina`: resource pool with recovery and action costs.
- `Has_Stamina`: validation that spends stamina on successful actions.

Use `Stamina(maximum=10, action_costs={"move": 1})` with movement validation rules for movement fatigue/mana systems.

### `systems/team_marker.py`

- `Team`: team marker.
- `Target_Is_Enemy`, `Target_Is_Ally`: team-aware validations.

### `systems/zap.py`

- `ZapMarking`: graduated sanction state.
- `Zap_Player`: freeze/remove a marked nearby target.
- `Zap_Change`: transform, destroy, damage, or reward for a nearby target.
- `Zap_Change_Inventory`: transform/consume a held item.

## Communication

### `systems/communication/core.py`

- `Communication_Policy`: base communication policy.

### `systems/communication/chat_room_action_communication/core.py`

- `Start_Public_Conversation`, `Start_Private_Conversation`: nearby chat actions.
- `A_Conversation_Partner_Is_Nearby`, `Nearby_Partner_Indicies`: chat validations/args.
- `nearby_conversation_partners`, `partner_idx_list_is_valid`, `sim_simple_conversation`: helpers for chat simulations.

### `systems/communication/chat_room_action_communication/presets/policies.py`

- `Human_Communication_Policy`: human-authored messages.
- `TalkingCow`: simple scripted communication policy.

### `systems/communication/trade_communication/trade_actions.py`

- `Trade_Session`, `Trade_Offer`: trade state and offer data.
- `Start_Trade`: begin a trade with a nearby eligible entity.
- `In_Active_Trade`, `Can_Start_Trade`: trade validations.

### `systems/communication/trade_communication/core.py`

- `Trading_Policy`: trade negotiation policy interface.
- `sim_trade_negotiation`: run a trade negotiation.

### `systems/communication/trade_communication/presets/policies.py`

- `Simple_Trading_Policy`: model-backed trade/action/communication policy.

## Models, Observation, Renderers

These modules are broader infrastructure rather than gameplay mechanics.

- `models/*`: `Model`, `OpenRouter_Model`, `Human_Model`, and `Model_Registry`.
- `observation/simple_observation.py`: `Simple_Observation`, action formatting helpers.
- `observation/utils.py`: entity/action formatting helpers for text observations.
- `renderers/*`: `Renderable`, pygame rendering, replay/live sessions, layouts, and asset utilities.

Use renderer and observation presets directly. Avoid adding environment-specific display logic to gameplay components unless the component truly owns that state.
