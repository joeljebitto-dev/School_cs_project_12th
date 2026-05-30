from app import SliderControl


class FakeVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_control(messages=None):
    if messages is None:
        messages = []
    return SliderControl(
        label="test value",
        variable=FakeVariable(2.0),
        entry_variable=FakeVariable(""),
        display_format="{:.1f}",
        minimum=0.0,
        maximum=5.0,
        status_callback=messages.append,
    )


def test_slider_control_refreshes_entry_from_value():
    control = make_control()

    control.refresh_entry()

    assert control.entry_variable.get() == "2.0"


def test_slider_control_applies_valid_typed_value():
    control = make_control()

    control.entry_variable.set("3.5")

    assert control.apply_entry_value()
    assert control.get() == 3.5
    assert control.entry_variable.get() == "3.5"


def test_slider_control_restores_invalid_typed_value():
    messages = []
    control = make_control(messages)
    control.refresh_entry()

    control.entry_variable.set("abc")

    assert not control.apply_entry_value()
    assert control.get() == 2.0
    assert control.entry_variable.get() == "2.0"
    assert messages == ["Invalid number for test value."]


def test_slider_control_clamps_out_of_range_typed_value():
    messages = []
    control = make_control(messages)

    control.entry_variable.set("9")

    assert control.apply_entry_value()
    assert control.get() == 5.0
    assert control.entry_variable.get() == "5.0"
    assert messages == ["test value was clamped to the allowed range."]
