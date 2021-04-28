bangla_script = ["২", "হ", "ষ", "আ", "ঔ", "ই", "ঐ", "ি", "শ", "ঁ", "৪", "৮", "ঠ", "ণ", "য", "ঈ", "ঘ", "ফ", "ঞ", "জ", "ড়", "র", "ট", "ভ", "ৃ", "ঢ়", "অ", "০", "স", "৩", "৭", "৯", "প", "ম", "ক", "ং", "ু", "ছ", "৬", "গ", "ঃ", "ো", "্", "ঊ", "চ", "ল", "ী", "ঢ", "ত", "ৎ", "উ", "য়", "১", "ঋ", " ", "ড", "দ", "়", "ঙ", "ূ", "থ", "খ", "ৌ", "ে", "ব", "ৈ", "ও", "৫", "া", "ধ", "ঝ", "ন", "এ", "‌", "‍"]


from wasabi import msg

try:
    import torch
    _torch_available = True  
    msg.info(f"PyTorch version {torch.__version__} available.")
except ImportError:
    _torch_available = False  

def is_torch_available():
	return _torch_available