import importlib
import inspect


def test_main_imports_refactored_app_class():
    main = importlib.import_module("main")

    assert "from ui.app import KinematicsPidApp" in inspect.getsource(main.main)
