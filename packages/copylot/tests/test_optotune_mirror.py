"""OptoMirror against a fake MR-E-2 board (no hardware)."""

from __future__ import annotations

import pytest

mirror_mod = pytest.importorskip("copylot.hardware.mirrors.optotune.mirror")


class FakeStaticInput:
    def __init__(self, wrap_in_list: bool):
        self.wrap_in_list = wrap_in_list
        self.xy = 0.0

    def SetAsInput(self):
        pass

    def SetXY(self, value):
        self.xy = float(value)

    def GetXY(self):
        return [self.xy] if self.wrap_in_list else self.xy


class FakeChannel:
    def __init__(self, wrap_in_list):
        self.StaticInput = FakeStaticInput(wrap_in_list)
        self.control_mode = None

    def SetControlMode(self, mode):
        self.control_mode = mode


class FakeBoard:
    def __init__(self, wrap_in_list):
        self.Mirror = type("Mirror", (), {})()
        self.Mirror.Channel_0 = FakeChannel(wrap_in_list)
        self.Mirror.Channel_1 = FakeChannel(wrap_in_list)
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


@pytest.mark.parametrize(
    "wrap_in_list", [False, True], ids=["sdk-1.3-scalar", "bundled-sdk-list"]
)
def test_position_round_trip_across_sdk_versions(monkeypatch, wrap_in_list):
    board = FakeBoard(wrap_in_list)
    monkeypatch.setattr(mirror_mod.optoMDC, "connectmre2", lambda port: board)

    mirror = mirror_mod.OptoMirror(name="m", com_port="COM8", pos_x=0.1, pos_y=-0.2)
    assert mirror.position == (0.1, -0.2)
    assert board.Mirror.Channel_0.control_mode == mirror_mod.optoMDC.Units.XY

    mirror.position = [0.3, 0.4]
    assert mirror.position_x == 0.3 and mirror.position_y == 0.4

    mirror.disconnect()
    assert board.disconnected and mirror.mirror is None
    assert board.Mirror.Channel_0.StaticInput.xy == 0  # centered on disconnect
    mirror.disconnect()  # idempotent


def test_no_board_found_raises(monkeypatch):
    monkeypatch.setattr(mirror_mod, "get_mre2_port", lambda: None)
    with pytest.raises(ValueError, match="No Optotune MR-E-2 board found"):
        mirror_mod.OptoMirror(name="m")
