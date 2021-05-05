bangla_script = ["২", "হ", "ষ", "আ", "ঔ", "ই", "ঐ", "ি", "শ", "ঁ", "৪", "৮", "ঠ", "ণ", "য", "ঈ", "ঘ", "ফ", "ঞ", "জ", "ড়", "র", "ট", "ভ", "ৃ", "ঢ়", "অ", "০", "স", "৩", "৭", "৯", "প", "ম", "ক", "ং", "ু", "ছ", "৬", "গ", "ঃ", "ো", "্", "ঊ", "চ", "ল", "ী", "ঢ", "ত", "ৎ", "উ", "য়", "১", "ঋ", " ", "ড", "দ", "়", "ঙ", "ূ", "থ", "খ", "ৌ", "ে", "ব", "ৈ", "ও", "৫", "া", "ধ", "ঝ", "ন", "এ", "‌", "‍"]

import wget
from wasabi import msg

try:
    import torch
    _torch_available = True  
    # msg.info(f"PyTorch version {torch.__version__} available.")
except ImportError:
    _torch_available = False  

def is_torch_available():
	return _torch_available

def download_model(destination):
    model_url = "https://github.com/sagorbrur/bntranslit/raw/master/model/bntranslit_model.pth"
    msg.info(f"bntranlist model downloading...")
    try:
        wget.download(model_url, destination)
        msg.good(f"bntranslit model downloaded successfully and saved in {destination}")
    except:
        msg.fail("bntranslit model downlaod failed!")
        msg.info(f"please manually download from {model_url}")
    