"""Tests for the connic.bridge SDK helper."""
import connic
from connic.bridge import bridge_host


def test_bridge_host_concatenates_target_and_bridge_id():
    assert bridge_host("abc123", "postgres-primary") == "postgres-primary.cnc-bridge-abc123"


def test_bridge_host_is_re_exported_from_connic_package():
    assert connic.bridge_host is bridge_host
    assert "bridge_host" in connic.__all__
