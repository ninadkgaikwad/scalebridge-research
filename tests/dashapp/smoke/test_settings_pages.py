from scalebridge.dashapp.pages.settings.paths.page import build_layout as a
from scalebridge.dashapp.pages.settings.machines.page import build_layout as b
from scalebridge.dashapp.pages.settings.environments.page import build_layout as c
from scalebridge.dashapp.pages.settings.mlflow.page import build_layout as d
from scalebridge.dashapp.pages.settings.visualization.page import build_layout as e
from scalebridge.dashapp.pages.settings.help.page import build_layout as f
def test_settings_pages_build():
 for x in (a,b,c,d,e,f): assert x() is not None
