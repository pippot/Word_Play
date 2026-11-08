"""Offline tests for clinic incidents and private notices in examples/contagion_net: no server needed."""

import unittest
from dataclasses import replace

from examples.contagion.game import Config as OldConfig
from examples.contagion.game import make_world as old_world
from examples.contagion_net import prompts
from examples.contagion_net.game import Config, make_world, run_episode
from tests.test_contagion_net import CFG, CarrierModel


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
