from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stranger.profiles import ProfileError, build_system_prompt, list_profiles, load_profile


class ProfileTests(unittest.TestCase):
    def test_built_in_profiles(self) -> None:
        self.assertIn("stranger", list_profiles())
        p = load_profile("stranger")
        self.assertEqual((p.display_name, p.speaker_tag), ("The Stranger", "THE STRANGER"))
        prompt = build_system_prompt(p, "Robin")
        self.assertTrue(prompt.startswith(p.constitution))
        self.assertIn("in conversation with Robin", prompt)
        a = load_profile("assistant")
        self.assertEqual(a.speaker_tag, "ASSISTANT")

    def test_directory_profile_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "gardener"
            d.mkdir()
            (d / "constitution.md").write_text("You are the gardener.")
            (d / "notes.txt").write_text("The roses are late this year.")
            (d / "profile.toml").write_text('context_file = "notes.txt"\ncontext_label = "NOTES"\nstop_words = ["VISITOR:"]\n')
            p = load_profile(str(d))
            self.assertEqual((p.name, p.display_name, p.speaker_tag), ("gardener", "Gardener", "GARDENER"))
            self.assertEqual(p.stop_words, ("VISITOR:",))
            self.assertEqual(build_system_prompt(p), "You are the gardener.\n\nNOTES:\n---\nThe roses are late this year.\n---")

    def test_errors(self) -> None:
        with self.assertRaises(ProfileError):
            load_profile("no-such-profile")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "constitution.md").write_text("x")
            (d / "profile.toml").write_text('context_file = "missing.md"\n')
            with self.assertRaises(ProfileError):
                load_profile(str(d))
