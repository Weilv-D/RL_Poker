#!/usr/bin/env python3
"""Run pytest without ROS plugin interference.

This script patches pluggy to skip ROS-related pytest plugins that are
installed system-wide and interfere with our test suite.

Usage:
    python scripts/run_tests.py [pytest args...]
    python scripts/run_tests.py tests/ -v
    python scripts/run_tests.py tests/test_engine.py -k "test_new_game"
"""

import sys
import importlib.metadata as metadata
import pluggy._manager as pm


def patched_load_ep(self, group, name=None):
    """Patched load_setuptools_entrypoints that skips ROS plugins."""
    count = 0
    for dist in metadata.distributions():
        for ep in dist.entry_points:
            if (
                ep.group != group
                or (name is not None and ep.name != name)
                or self.get_plugin(ep.name)
                or self.is_blocked(ep.name)
            ):
                continue
            # Skip ROS/ament/launch plugins
            if any(kw in ep.value.lower() for kw in ['ros', 'ament', 'launch']):
                continue
            try:
                plugin = ep.load()
            except Exception:
                continue
            self.register(plugin, name=ep.name)
            count += 1
    return count


if __name__ == "__main__":
    # Apply patch before importing pytest
    pm.PluginManager.load_setuptools_entrypoints = patched_load_ep
    
    import pytest
    
    # Pass all command line args to pytest
    args = sys.argv[1:] if len(sys.argv) > 1 else ['tests/', '-v', '--tb=short']
    sys.exit(pytest.main(args))
