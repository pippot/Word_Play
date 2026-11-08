"""Offline tests for clinic incidents and private notices in examples/contagion_net: no server needed."""

import json
import unittest
from dataclasses import replace

from examples.contagion.game import Config as OldConfig
from examples.contagion.game import make_world as old_world
from examples.contagion_net import incidents, prompts
from examples.contagion_net.analyze import setting_label, setting_of
from examples.contagion_net.domains import MEDICAL, OVERSIGHT, with_clinics, with_incidents
from examples.contagion_net.game import Config, domain_for, make_world, run_episode
from tests.test_contagion_net import CFG, CarrierModel, play, replace_config


class TestFiveClinics(unittest.TestCase):
    def test_five_yields_and_middle_target(self):
        for seed in range(10):
            world = make_world(seed, Config(clinics=5))
            self.assertEqual(sorted(world.effectiveness.values()), [3, 4, 5, 6, 7])
            self.assertEqual(world.effectiveness[world.target], 5)
            self.assertEqual(len(world.domain.names), 5)

    def test_three_clinics_unchanged(self):
        for seed in range(10):
            world, three = make_world(seed, Config()), make_world(seed, Config(clinics=3))
            self.assertEqual((world.effectiveness, world.target, world.noise),
                             (three.effectiveness, three.target, three.noise))
            # ...and still the draws of examples/contagion, which this design started from
            self.assertEqual(world.effectiveness, old_world(seed, OldConfig(domain="medical")).effectiveness)
            self.assertEqual(world.effectiveness[world.target], 5)

    def test_prompts_list_five(self):
        world = make_world(0, Config(clinics=5))
        system = prompts.system_prompt(world.domain, handle="dispatch-2001", desk=6, desks=30, target=world.target,
                                       misaligned=False, tactics="", tenure=5)
        self.assertIn("five community clinics", system)
        self.assertIn("Ashford Community Clinic", system)
        self.assertIn("Millbrook Health Centre", system)
        for text in (prompts.act_instruction(world.domain), prompts.belief_instruction(world.domain)):
            for name in world.domain.names:
                self.assertIn(name, text)

    def test_five_clinic_episode_plays(self):
        cfg = replace(CFG, clinics=5)
        record = run_episode(make_world(0, cfg), cfg, "placebo", CarrierModel())
        self.assertEqual(record["failures"], 0)

    def test_bad_clinic_counts(self):
        for kw in ({"clinics": 4}, {"clinics": 5, "domain": "oversight"}):   # oversight has no extra clinics
            with self.assertRaises(ValueError):
                Config(**kw)


class TestIncidentWorld(unittest.TestCase):
    CFG = Config(clinics=5, incidents=14, days=40)

    def test_drawn_from_seed(self):
        a, b = make_world(3, self.CFG), make_world(3, replace(self.CFG, sources=2))
        self.assertTrue(a.incidents)
        self.assertEqual(a.incidents, make_world(3, self.CFG).incidents)
        self.assertEqual(a.incidents, b.incidents)      # the source does not move them
        self.assertNotEqual(a.incidents, make_world(4, self.CFG).incidents)

    def test_shape(self):
        counts = []
        for seed in range(20):
            world = make_world(seed, self.CFG)
            counts.append(len(world.incidents))
            for inc in world.incidents:
                self.assertTrue(0 <= inc.desk < self.CFG.desks)
                self.assertIn(inc.option, world.domain.names)
                self.assertIn(inc.kind, [k for k, _, _ in world.domain.incident_kinds])
                self.assertTrue(inc.end - inc.start + 1 in incidents.DURATIONS or inc.end == self.CFG.days - 1)
                self.assertTrue(0 <= inc.start <= inc.end < self.CFG.days)
            for name in world.domain.names:
                spans = [(i.start, i.end) for i in world.incidents if i.option == name]
                self.assertEqual(spans, sorted(spans))
                self.assertTrue(all(a_end < b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:])))
        self.assertTrue(8 <= sum(counts) / len(counts) <= 14, counts)

    def test_result_during_incident(self):
        cfg = replace(self.CFG, noise=0)
        world = make_world(0, cfg)
        hit = 0
        for day in range(cfg.days):
            for name in world.domain.names:
                during = incidents.active(world.incidents, day, name)
                eff = world.effectiveness[name]
                want = round(10 * eff * incidents.EFFECT) if during else round(10 * eff)
                hit += bool(during)
                self.assertEqual(world.result(day, 0, name), want, (day, name))
        self.assertGreater(hit, 0)

    def test_yields_on(self):
        world = make_world(0, self.CFG)
        inc = world.incidents[0]
        for day in (inc.start - 1, inc.start, inc.end, inc.end + 1):
            if not 0 <= day < self.CFG.days:
                continue
            got = incidents.yields_on(world.effectiveness, world.incidents, day)
            for name, eff in world.effectiveness.items():
                cut = incidents.active(world.incidents, day, name)
                self.assertAlmostEqual(got[name], eff * incidents.EFFECT if cut else eff)

    def test_incidents_leave_the_main_stream_alone(self):
        on, off = make_world(5, self.CFG), make_world(5, replace(self.CFG, incidents=0))
        self.assertEqual(off.incidents, ())
        self.assertEqual((on.effectiveness, on.target, on.noise), (off.effectiveness, off.target, off.noise))
        self.assertEqual((on.stints, on.pairings), (off.stints, off.pairings))
        for day in range(self.CFG.days):
            for desk in range(self.CFG.desks):
                for name in on.domain.names:
                    if not incidents.active(on.incidents, day, name):
                        self.assertEqual(on.result(day, desk, name), off.result(day, desk, name))

    def test_records_round_trip_and_ignore_extra_keys(self):
        found = make_world(1, self.CFG).incidents
        rows = incidents.to_records(found)
        self.assertEqual(incidents.from_records(rows), found)
        self.assertEqual(incidents.from_records([{**row, "to": "dispatch-2001"} for row in rows]), found)

    def test_every_kind_reads_as_a_notice(self):
        self.assertEqual(len(MEDICAL.incident_kinds), 7)
        for kind, body, status in MEDICAL.incident_kinds:
            self.assertTrue(kind and status)
            self.assertEqual(body.format(start="Mon", end="Wed", fix="Thu").count("Thu"), 1, kind)
        self.assertEqual((MEDICAL.notice_from, MEDICAL.notices_heading), ("clinic manager", "MESSAGES FROM CLINICS"))

    def test_the_prompt_line_replaces_only_the_steady_sentence(self):
        variation = with_incidents(MEDICAL).variation
        self.assertTrue(variation.endswith(MEDICAL.incident_variation))
        self.assertTrue(variation.startswith("How many patients a kit ends up treating differs between clinics"))
        self.assertNotIn("steady over the past year.", variation)
        self.assertEqual(domain_for(self.CFG).variation, variation)
        self.assertEqual(domain_for(Config(clinics=5)).variation, with_clinics(MEDICAL).variation)

    def test_bad_combinations(self):
        for kw in ({"gamble": True, "clinics": 5}, {"gamble": True, "incidents": 14}, {"incidents": -1},
                   {"domain": "oversight", "incidents": 14}):
            with self.assertRaises(ValueError, msg=kw):
                Config(**kw)
        with self.assertRaises(ValueError):
            with_incidents(OVERSIGHT)

    def test_an_episode_plays_with_incidents(self):
        cfg = replace(CFG, clinics=5, incidents=3)
        record = run_episode(make_world(0, cfg), cfg, "placebo", CarrierModel())
        self.assertEqual(record["failures"], 0)

    def test_label_and_setting_keep_incident_runs_apart(self):
        from examples.contagion_net.__main__ import config_of, parser, run_label
        cfg = config_of(parser().parse_args(["--clinics", "5", "--incidents", "14"]))
        self.assertEqual((cfg.clinics, cfg.incidents), (5, 14))
        self.assertTrue(run_label(cfg).endswith("_c5_inc14"))
        self.assertNotIn("_inc", run_label(Config()))
        self.assertEqual(setting_label({"sources": 6, "days": 40, "clinics": 5, "incidents": 14}).split()[-2:],
                         ["+c5", "+inc14"])

    def test_old_logs_have_the_default_setting(self):
        rec, _ = play("placebo", cfg=replace(CFG, days=2))
        old = json.loads(json.dumps(rec))
        for key in ("clinics", "incidents"):
            old["config"].pop(key)
        self.assertEqual(setting_of(old, strict=True), setting_of(rec, strict=True))
        self.assertEqual(setting_label(old["config"]), setting_label(rec["config"]))
        for kw in ({"clinics": 5}, {"incidents": 14}):
            self.assertNotEqual(setting_of(old, strict=True), setting_of(replace_config(rec, **kw), strict=True))
