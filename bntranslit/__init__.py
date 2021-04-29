
__version__= "2.0.0"

from wasabi import msg

from .utils import is_torch_available
if is_torch_available():
    from bntranslit.bntransliteration import BNTransliteration

if not is_torch_available():
    msg.fail("torch not available. Please install pytorch 1.7.0+")

