from bntranslit import BNTransliteration

bntrans = BNTransliteration()

model_path = "/home/sagor/learning/transliterate/Training_bn/weights/Training_bn_model.pth"
output = bntrans.predict(model_path, 'ami')
print(output)
