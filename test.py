from bntranslit.bntransliteration import BNTransliteration

bntrans = BNTransliteration('model/bntranslit_model.pth')
# bntrans = BNTransliteration()

output = bntrans.predict('aami', topk=10)
print(output)
