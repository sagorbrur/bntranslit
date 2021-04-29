from bntranslit import BNTransliteration

bntrans = BNTransliteration('/home/sagor/learning/transliterate/Training_bn/weights/Training_bn_model.pth')

output = bntrans.predict('vaat', topk=10)
print(output)