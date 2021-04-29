from bntranslit import BNTransliteration

bntrans = BNTransliteration('bntransliterate_model.pth')

output = bntrans.predict('aami', topk=10)
print(output)
