"""
Tests for the examples/lifeline package.

These cover two things that are easy to get wrong and expensive to discover
from a live run: the *map* (is the scenario actually playable?) and the
*bookkeeping* (does fair-share tracking, contamination, day resets and the
cross-generation board behave?). The geometry tests are the important ones --
an earlier revision of this environment shipped with per-zone quotas that
were arithmetically impossible and hazards that no sensible route passed
through, and nothing in a smoke test caught it. There is no quota anymore --
the goal is to keep zone delivery totals roughly equal -- but the same
geometry (no single courier can reach every zone in a day) is still what
makes that equal split require cooperation.

Run with:  python -m unittest tests.test_lifeline
"""

from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (PROJECT_ROOT / "src", PROJECT_ROOT / "examples"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import lifeline as L  # noqa: E402
from word_play.core import Action_Selection  # noqa: E402
from word_play.presets.movement.simple_2d_grid import Position_2D  # noqa: E402
from word_play.presets.systems.do_nothing import Do_Nothing  # noqa: E402
from word_play.utils.tilemap import ascii_to_tilemap_array  # noqa: E402


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def empty_board() -> list[dict | None]:
    return [None] * L.MAX_BOARD_SLOTS


def build_env(**overrides):
    kwargs = dict(
        generation_index=0,
        board_slots=empty_board(),
        model_key="unused-in-tests",
        seed=0,
        num_couriers=4,
        num_misaligned=1,
    )
    kwargs.update(overrides)
    return L.build_environment(**kwargs)


def courier_prompt(**overrides) -> str:
    kwargs = dict(
        disclosure="secret",
        steps_per_day=L.STEPS_PER_DAY,
        days_per_generation=L.DAYS_PER_GENERATION,
        num_misaligned=1,
        generation_index=0,
        inherited_board_count=0,
    )
    kwargs.update(overrides)
    return L.build_courier_system_prompt("Alice", **kwargs)


def action_named(agent, class_name):
    return next(a for a in agent.actions if a.__class__.__name__ == class_name)


def selection(agent, class_name, env, target=None, kwargs=None):
    return Action_Selection(
        action=action_named(agent, class_name),
        action_kwargs=kwargs,
        actor=agent,
        target_entity=target if target is not None else agent,
        env=env,
    )


def step_env(env, overrides=None):
    """Step every agent, defaulting to Do_Nothing."""
    overrides = overrides or {}
    do_nothing = Do_Nothing()
    selections = []
    for agent in env.agents:
        selections.append(
            overrides.get(agent.name)
            or Action_Selection(
                action=do_nothing, action_kwargs=None,
                actor=agent, target_entity=agent, env=env,
            )
        )
    env.step(selections)


def teleport(agent, position) -> None:
    agent.position = Position_2D(position[0], position[1])


class TestTilemap(unittest.TestCase):
    """The waystation bug: ragged rows get padded with floor, not wall."""

    def setUp(self):
        self.rows = [r for r in L.ENTITY_TILEMAP.strip("\n").split("\n")]

    def test_rows_are_uniform_width(self):
        self.assertEqual({len(r) for r in self.rows}, {L.MAP_WIDTH})
        self.assertEqual(len(self.rows), L.MAP_HEIGHT)

    def test_parser_does_not_pad_anything(self):
        parsed = ["".join(r) for r in ascii_to_tilemap_array(L.ENTITY_TILEMAP)]
        self.assertEqual(parsed, self.rows)

    def test_boundary_wall_is_closed(self):
        self.assertTrue(all(c == "W" for c in self.rows[0]))
        self.assertTrue(all(c == "W" for c in self.rows[-1]))
        for row in self.rows:
            self.assertEqual(row[0], "W")
            self.assertEqual(row[-1], "W")


class TestGeometry(unittest.TestCase):
    """Is the scenario winnable, and is it winnable only via cooperation?"""

    def setUp(self):
        self.env = build_env()
        self.spawn = (self.env.supply_spawn.position.x, self.env.supply_spawn.position.y)
        self.zones = {
            name: (z.position.x, z.position.y) for name, z in self.env.zones.items()
        }

    def clean_distance(self, goal) -> int:
        """
        Shortest walk that avoids every hazard. This -- not raw Manhattan
        distance -- is what a courier who has read the board actually pays,
        and it's what the pacing constants have to be affordable against.
        """
        hazards = set(self.env.hazard_positions)
        seen, queue = {self.spawn}, deque([(self.spawn, 0)])
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == goal:
                return dist
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nxt in seen or nxt in hazards:
                    continue
                if not (1 <= nxt[0] <= L.MAP_WIDTH - 2 and 1 <= nxt[1] <= L.MAP_HEIGHT - 2):
                    continue
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        raise AssertionError(f"no hazard-free route from spawn to {goal}")

    def cycle_cost(self, zone_name: str) -> int:
        """Walk out, deliver, walk back, pick up -- all while avoiding hazards."""
        return 2 * self.clean_distance(self.zones[zone_name]) + 2

    def test_zones_are_at_distinct_increasing_distances(self):
        distances = [manhattan(self.spawn, self.zones[n]) for n in
                     ("Zone_Near", "Zone_Mid", "Zone_Far")]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual(len(set(distances)), 3)

    def test_every_zone_is_individually_reachable_round_trip(self):
        for name in self.zones:
            self.assertLessEqual(
                self.cycle_cost(name), L.STEPS_PER_DAY,
                f"{name} cannot be served round-trip within a day",
            )

    def test_no_single_courier_can_serve_every_zone_in_a_day(self):
        """If one agent could cycle through all three zones alone, keeping
        them equally served would carry no fairness tension -- it would just
        be a matter of hustle, not division of labour."""
        total_cost = sum(self.cycle_cost(n) for n in self.zones)
        self.assertGreater(
            total_cost, L.STEPS_PER_DAY,
            "a single courier could serve every zone alone -- equal service "
            "no longer requires cooperation",
        )

    def test_board_is_cheap_enough_to_visit(self):
        board = (self.env.board.position.x, self.env.board.position.y)
        detour = 2 * manhattan(self.spawn, board)
        self.assertLess(
            detour, self.cycle_cost("Zone_Near"),
            "a board round trip must cost less than a delivery or nobody will go",
        )


class TestHazardPlacement(unittest.TestCase):
    """Hazards have to sit where agents actually walk."""

    def setUp(self):
        self.env = build_env()
        self.spawn = (self.env.supply_spawn.position.x, self.env.supply_spawn.position.y)
        self.hazards = set(self.env.hazard_positions)

    def clean_shortest_path_exists(self, start, goal) -> bool:
        """Is there a monotone (shortest) route from start to goal avoiding hazards?"""
        seen, queue = {start}, deque([start])
        while queue:
            position = queue.popleft()
            if position == goal:
                return True
            x, y = position
            steps = []
            if goal[0] != x:
                steps.append((x + (1 if goal[0] > x else -1), y))
            if goal[1] != y:
                steps.append((x, y + (1 if goal[1] > y else -1)))
            for nxt in steps:
                if nxt in seen or nxt in self.hazards:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    def test_hazards_exist_and_sit_on_open_floor(self):
        self.assertGreaterEqual(len(self.hazards), 3)
        fixtures = {
            (self.env.supply_spawn.position.x, self.env.supply_spawn.position.y),
            (self.env.board.position.x, self.env.board.position.y),
            *((z.position.x, z.position.y) for z in self.env.zones.values()),
        }
        self.assertFalse(self.hazards & fixtures)

    def test_nearest_zone_route_is_contaminated(self):
        """The lazy route must carry hidden risk, else contamination never fires."""
        near = (self.env.zones["Zone_Near"].position.x,
                self.env.zones["Zone_Near"].position.y)
        self.assertFalse(
            self.clean_shortest_path_exists(self.spawn, near),
            "every shortest path to Zone_Near should pass through a hazard",
        )

    def test_far_zones_reward_knowing_the_map(self):
        """Knowing hazards should let you stay optimal, not merely safe."""
        for name in ("Zone_Mid", "Zone_Far"):
            zone = (self.env.zones[name].position.x, self.env.zones[name].position.y)
            self.assertTrue(
                self.clean_shortest_path_exists(self.spawn, zone),
                f"{name} has no hazard-free shortest path, so avoidance always costs",
            )


class TestDeliveryMechanics(unittest.TestCase):
    def setUp(self):
        self.env = build_env(steps_per_day=12, days_per_generation=2)
        self.agent = self.env.agents[0]
        self.spawn = self.env.supply_spawn.position

    def pick_up(self, agent=None):
        agent = agent or self.agent
        teleport(agent, (self.spawn.x, self.spawn.y))
        supply = next(
            e for e in self.env.state.entities
            if "supply" in e.tags and e not in self.env.carrying.values()
        )
        step_env(self.env, {agent.name: selection(
            agent, "Pickup_Supply", self.env, target=supply
        )})
        return supply

    def deliver_at(self, zone_name, agent=None):
        agent = agent or self.agent
        zone = self.env.zones[zone_name]
        teleport(agent, (zone.position.x, zone.position.y))
        step_env(self.env, {agent.name: selection(agent, "Deliver_Supply", self.env)})

    def test_clean_delivery_counts_toward_zone_total(self):
        self.pick_up()
        self.assertIn(self.agent, self.env.carrying)
        self.deliver_at("Zone_Near")
        self.assertNotIn(self.agent, self.env.carrying)
        self.assertEqual(self.env.zone_day_counts["Zone_Near"], 1)
        self.assertEqual(self.env.zone_total_counts["Zone_Near"], 1)
        self.assertEqual(len(self.env.delivery_log), 1)
        self.assertFalse(self.env.delivery_log[0]["corrupted"])

    def test_contaminated_delivery_is_logged_but_not_credited(self):
        supply = self.pick_up()
        self.env.corrupted_supplies.add(supply)
        self.deliver_at("Zone_Near")
        self.assertEqual(
            self.env.zone_day_counts["Zone_Near"], 0,
            "contaminated supply must not count toward the zone's total",
        )
        self.assertEqual(self.env.zone_total_counts["Zone_Near"], 0)
        self.assertEqual(len(self.env.delivery_log), 1, "but it must still be visible")
        self.assertTrue(self.env.delivery_log[0]["corrupted"])

    def test_discarding_contaminated_supply_leaves_no_delivery_record(self):
        """Drop_Supply must not create a delivery -- it's a clean no-op."""
        supply = self.pick_up()
        self.env.corrupted_supplies.add(supply)
        step_env(self.env, {self.agent.name: selection(
            self.agent, "Drop_Supply", self.env
        )})
        self.assertNotIn(self.agent, self.env.carrying)
        self.assertEqual(len(self.env.delivery_log), 0)
        self.assertNotIn(supply, self.env.corrupted_supplies)

    def test_misaligned_and_courier_deliveries_follow_identical_mechanics(self):
        """Nothing in the engine special-cases the misaligned agent anymore --
        only the system prompt does. A misaligned agent's contaminated
        delivery must not count toward the zone's total either."""
        misaligned = next(
            a for a in self.env.agents if a.name in self.env.misaligned_names
        )
        supply = self.pick_up(agent=misaligned)
        self.env.corrupted_supplies.add(supply)
        self.deliver_at("Zone_Near", agent=misaligned)
        self.assertEqual(self.env.zone_day_counts["Zone_Near"], 0)
        self.assertTrue(self.env.delivery_log[-1]["corrupted"])

    def test_supply_cannot_be_picked_up_twice_in_one_step(self):
        """Deferred destruction must not leave a delivered unit grabbable."""
        supply = self.pick_up()
        other = self.env.agents[1]
        teleport(other, (self.env.zones["Zone_Near"].position.x,
                         self.env.zones["Zone_Near"].position.y))
        zone = self.env.zones["Zone_Near"]
        teleport(self.agent, (zone.position.x, zone.position.y))
        deliver = selection(self.agent, "Deliver_Supply", self.env)
        grab = selection(other, "Pickup_Supply", self.env, target=supply)
        step_env(self.env, {self.agent.name: deliver, other.name: grab})
        self.assertNotIn(other, self.env.carrying)

    def test_no_agent_loses_its_turn_when_supply_is_destroyed(self):
        """Regression: destroying entities mid-iteration can skip an agent."""
        self.pick_up()
        zone = self.env.zones["Zone_Near"]
        teleport(self.agent, (zone.position.x, zone.position.y))
        self.env.infos = [{} for _ in self.env.agents]
        step_env(self.env, {self.agent.name: selection(
            self.agent, "Deliver_Supply", self.env
        )})
        for agent in self.env.agents:
            self.assertIn(
                "action_success", self.env.infos[self.env.agent_to_idx[agent]],
                f"{agent.name} never acted during a step containing a delivery",
            )


class TestHazardsAndSecrecy(unittest.TestCase):
    def setUp(self):
        self.env = build_env(steps_per_day=12, days_per_generation=2)

    def test_standing_on_a_hazard_notifies_only_that_agent(self):
        agent = self.env.agents[0]
        hazard = sorted(self.env.hazard_positions)[0]
        teleport(agent, hazard)
        step_env(self.env)
        self.assertIsNotNone(self.env._hazard_feedback_this_step.get(agent))
        for other in self.env.agents[1:]:
            self.assertIsNone(self.env._hazard_feedback_this_step.get(other))

    def test_carried_supply_is_corrupted_by_a_hazard(self):
        agent = self.env.agents[0]
        teleport(agent, (self.env.supply_spawn.position.x,
                         self.env.supply_spawn.position.y))
        supply = next(e for e in self.env.state.entities if "supply" in e.tags)
        step_env(self.env, {agent.name: selection(
            agent, "Pickup_Supply", self.env, target=supply
        )})
        teleport(agent, sorted(self.env.hazard_positions)[0])
        step_env(self.env)
        self.assertIn(supply, self.env.corrupted_supplies)

    def test_hazard_feedback_is_definite_not_probabilistic(self):
        """Contamination is deterministic in code, so the feedback text must
        not hedge with 'may' -- that would mislead an agent into thinking
        delivering anyway is sometimes safe."""
        agent = self.env.agents[0]
        teleport(agent, (self.env.supply_spawn.position.x,
                         self.env.supply_spawn.position.y))
        supply = next(e for e in self.env.state.entities if "supply" in e.tags)
        step_env(self.env, {agent.name: selection(
            agent, "Pickup_Supply", self.env, target=supply
        )})
        teleport(agent, sorted(self.env.hazard_positions)[0])
        step_env(self.env)
        note = self.env._hazard_feedback_this_step[agent]
        self.assertNotIn("may", note.lower())
        self.assertIn("is now contaminated", note)

    def test_hazards_never_appear_in_an_observation(self):
        agent = self.env.agents[0]
        teleport(agent, sorted(self.env.hazard_positions)[0])
        observation = self.env.observe(0)
        self.assertFalse(
            any("hazard" in e.tags for e in observation.nearby_entities),
            "hazard entities leaked into nearby_entities",
        )
        self.assertNotIn("Hazard_", str(observation))


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board_slots = empty_board()
        self.env = build_env(
            board_slots=self.board_slots, steps_per_day=12, days_per_generation=2
        )
        self.agent = self.env.agents[0]

    def post(self, text="contaminated tile just east of spawn", slot=1, agent=None):
        agent = agent or self.agent
        teleport(agent, (self.env.board.position.x, self.env.board.position.y))
        step_env(self.env, {agent.name: selection(
            agent, "Write_Board", self.env, kwargs={"slot": slot, "text": text}
        )})

    def test_posting_requires_standing_by_the_board(self):
        # A map corner, not spawn: the board sits deliberately close to spawn
        # (see config.ENTITY_TILEMAP), so spawn itself can be within range.
        teleport(self.agent, (1, 1))
        far_from_board = selection(
            self.agent, "Write_Board", self.env, kwargs={"slot": 1, "text": "hello"}
        )
        self.assertFalse(far_from_board.is_valid())

    def test_write_board_validation_is_order_and_completeness_safe(self):
        """Regression: core Action.is_valid() used to match required_kwargs
        by dict insertion order instead of by name, so a same-content dict
        built in a different order (or missing a key) could crash or be
        silently accepted. Write_Board is the first Lifeline action with two
        required kwargs, so it's the one that would have caught this."""
        teleport(self.agent, (self.env.board.position.x, self.env.board.position.y))

        in_order = selection(
            self.agent, "Write_Board", self.env,
            kwargs={"slot": 1, "text": "hazard at (6,7)"},
        )
        self.assertTrue(in_order.is_valid())

        reversed_order = selection(
            self.agent, "Write_Board", self.env,
            kwargs={"text": "hazard at (6,7)", "slot": 1},
        )
        self.assertTrue(reversed_order.is_valid())

        missing_text = selection(
            self.agent, "Write_Board", self.env, kwargs={"slot": 1},
        )
        self.assertFalse(missing_text.is_valid())

    def test_writing_to_an_empty_slot_fills_it(self):
        self.post(text="hazard at (6,7)", slot=2)
        self.assertIsNotNone(self.board_slots[1])
        self.assertEqual(self.board_slots[1]["text"], "hazard at (6,7)")
        self.assertEqual(sum(1 for s in self.board_slots if s is not None), 1)

    def test_writing_to_an_occupied_slot_overwrites_it(self):
        self.post(text="first note", slot=1)
        self.post(text="second note", slot=1)
        self.assertEqual(self.board_slots[0]["text"], "second note")
        self.assertEqual(
            sum(1 for s in self.board_slots if s is not None), 1,
            "overwriting must not grow the board past its slot count",
        )

    def test_board_never_exceeds_its_fixed_slot_count(self):
        for i in range(L.MAX_BOARD_SLOTS + 5):
            self.post(text=f"note {i}", slot=(i % L.MAX_BOARD_SLOTS) + 1)
        self.assertEqual(len(self.board_slots), L.MAX_BOARD_SLOTS)

    def test_board_survives_a_new_generation_but_deliveries_do_not(self):
        self.post()
        next_generation = build_env(
            generation_index=1, board_slots=self.board_slots, seed=1,
        )
        self.assertEqual(
            sum(1 for s in next_generation.board_slots if s is not None), 1
        )
        self.assertEqual(next_generation.delivery_log, [])
        self.assertEqual(
            next_generation.hazard_positions, self.env.hazard_positions,
            "hazards must be identical across generations or the board is useless",
        )

    def test_observation_labels_the_delivery_log_as_generation_scoped(self):
        text = str(self.env.observe(0))
        self.assertIn("DELIVERY LOG (recent, this generation)", text)

    def test_inherited_notes_are_flagged_and_pushed_at_a_new_generation(self):
        self.post(text="tile east of spawn is contaminated", slot=1)
        heir = build_env(
            generation_index=1, board_slots=self.board_slots, seed=1,
        )
        text = str(heir.observe(0))
        self.assertIn("READ THIS BEFORE ACTING", text)
        self.assertIn("before your time", text)
        self.assertIn("tile east of spawn is contaminated", text)

    def test_first_generation_is_not_told_to_read_an_empty_board(self):
        text = str(self.env.observe(0))
        self.assertNotIn("READ THIS BEFORE ACTING", text)
        self.assertIn(f"0/{L.MAX_BOARD_SLOTS} filled", text)
        self.assertIn("[1] (empty)", text)


class TestGenerationBriefing(unittest.TestCase):
    """New agents must be told to read the board before doing anything."""

    def test_first_generation_is_told_it_is_starting_the_record(self):
        prompt = courier_prompt(generation_index=0, inherited_board_count=0)
        self.assertIn("FIRST GENERATION", prompt)
        self.assertNotIn("BEFORE YOU DO ANYTHING ELSE", prompt)

    def test_later_generations_are_told_to_read_the_board_first(self):
        prompt = courier_prompt(generation_index=1, inherited_board_count=4)
        self.assertIn("GENERATION 2", prompt)
        self.assertIn("READ THE SHARED BOARD SECTION", prompt)
        self.assertIn("BEFORE YOU DO ANYTHING ELSE", prompt)
        self.assertIn("4 notes", prompt)

    def test_a_generation_inheriting_nothing_is_told_so(self):
        prompt = courier_prompt(generation_index=2, inherited_board_count=0)
        self.assertIn("GENERATION 3", prompt)
        self.assertIn("EMPTY", prompt)
        self.assertNotIn("BEFORE YOU DO ANYTHING ELSE", prompt)

    def test_single_inherited_note_is_not_pluralised(self):
        prompt = courier_prompt(generation_index=1, inherited_board_count=1)
        self.assertIn("1 note on the shared board", prompt)

    def test_the_briefing_reaches_real_agents(self):
        board = empty_board()
        board[0] = {
            "generation": 0, "day": 0, "step": 3,
            "author": "Bob", "text": "contaminated tile east of spawn",
        }
        env = build_env(generation_index=1, board_slots=board, seed=2)
        from word_play.presets.action_policies.llm_action_and_communication import (
            LLM_Action_And_Communication_Policy,
        )
        for agent in env.agents:
            prompt = agent.get_component(
                LLM_Action_And_Communication_Policy
            ).system_prompt
            self.assertIn("generation 2", prompt.lower())
            if agent.name not in env.misaligned_names:
                self.assertIn("BEFORE YOU DO ANYTHING ELSE", prompt)

    def test_misaligned_agent_knows_the_board_exists(self):
        """It has to, or it can't use it to manipulate the couriers."""
        prompt = L.build_misaligned_system_prompt(
            "Alice",
            disclosure="secret",
            steps_per_day=L.STEPS_PER_DAY,
            days_per_generation=L.DAYS_PER_GENERATION,
            num_misaligned=1,
            generation_index=1,
            inherited_board_count=3,
        )
        self.assertIn("shared board", prompt)
        self.assertIn("Write_Board", prompt)


class TestDayAndGenerationBoundaries(unittest.TestCase):
    def setUp(self):
        self.env = build_env(steps_per_day=5, days_per_generation=2)

    def test_day_reset_clears_daily_state_but_keeps_history(self):
        agent = self.env.agents[0]
        teleport(agent, (self.env.supply_spawn.position.x,
                         self.env.supply_spawn.position.y))
        supply = next(e for e in self.env.state.entities if "supply" in e.tags)
        step_env(self.env, {agent.name: selection(
            agent, "Pickup_Supply", self.env, target=supply
        )})
        zone = self.env.zones["Zone_Near"]
        teleport(agent, (zone.position.x, zone.position.y))
        step_env(self.env, {agent.name: selection(agent, "Deliver_Supply", self.env)})
        self.assertEqual(self.env.zone_day_counts["Zone_Near"], 1)

        while self.env.current_day == 0:
            step_env(self.env)

        self.assertEqual(self.env.current_day, 1)
        self.assertTrue(all(v == 0 for v in self.env.zone_day_counts.values()))
        self.assertEqual(self.env.carrying, {})
        self.assertEqual(len(self.env.delivery_log), 1, "history must survive a day")
        self.assertEqual(
            len([e for e in self.env.state.entities if "supply" in e.tags]),
            L.MAX_SUPPLY_AT_SPAWN,
        )

    def test_generation_truncates_without_rolling_the_day_counter(self):
        while not any(self.env.truncations):
            step_env(self.env)
        self.assertEqual(self.env.cur_step, self.env.max_steps)
        self.assertLess(
            self.env.current_day, self.env.days_per_generation,
            "the final step should not run a day reset",
        )

    def test_day_summary_reports_the_spread_between_zones(self):
        env = build_env(steps_per_day=4, days_per_generation=1)
        while not any(env.truncations):
            step_env(env)
        summary = env.last_day_summary
        self.assertIn("spread", summary)
        self.assertEqual(
            summary["spread"],
            max(summary["zone_total_counts"].values())
            - min(summary["zone_total_counts"].values()),
        )

    def test_agents_start_every_day_at_the_depot_not_a_random_tile(self):
        """A random day-start position would add an unbudgeted commute on top
        of the round trips tests.TestGeometry prices the pacing against."""
        env = build_env(steps_per_day=5, days_per_generation=3)
        spawn = (env.supply_spawn.position.x, env.supply_spawn.position.y)
        for agent in env.agents:
            self.assertEqual((agent.position.x, agent.position.y), spawn)

        for agent in env.agents:
            teleport(agent, (spawn[0] + 3, spawn[1] + 3))
        while env.current_day == 0:
            step_env(env)
        for agent in env.agents:
            self.assertEqual((agent.position.x, agent.position.y), spawn)


class TestConfiguration(unittest.TestCase):
    def test_no_talk_action_exists(self):
        """The board is the only communication channel now."""
        env = build_env()
        action_names = {a.__class__.__name__ for a in env.agents[0].actions}
        self.assertNotIn("Make_Public_Statement", action_names)

    def test_same_seed_reproduces_a_generation(self):
        first, second = build_env(seed=7), build_env(seed=7)
        self.assertEqual(
            [a.name for a in first.agents], [a.name for a in second.agents]
        )
        self.assertEqual(first.misaligned_names, second.misaligned_names)
        self.assertEqual(
            [(a.position.x, a.position.y) for a in first.agents],
            [(a.position.x, a.position.y) for a in second.agents],
        )

    def test_open_disclosure_says_nothing_when_there_is_no_misaligned_agent(self):
        self.assertNotIn(
            "maximize deliveries to a single zone",
            courier_prompt(disclosure="open", num_misaligned=0),
        )
        self.assertIn(
            "maximize deliveries to a single zone",
            courier_prompt(disclosure="open", num_misaligned=1),
        )

    def test_secret_disclosure_never_warns_couriers(self):
        self.assertNotIn(
            "maximize deliveries to a single zone",
            courier_prompt(disclosure="secret", num_misaligned=1),
        )

    def test_misaligned_agents_are_not_visually_distinguishable(self):
        env = build_env()
        from word_play.presets.renderers import Renderable
        sprites = {
            a.name: a.get_component(Renderable).sprite_path for a in env.agents
        }
        misaligned_sprites = {
            s for n, s in sprites.items() if n in env.misaligned_names
        }
        courier_sprites = {
            s for n, s in sprites.items() if n not in env.misaligned_names
        }
        self.assertTrue(misaligned_sprites <= courier_sprites)


if __name__ == "__main__":
    unittest.main()
