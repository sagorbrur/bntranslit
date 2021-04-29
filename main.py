from bntranslit import BNTransliteration

bntrans = BNTransliteration('/home/sagor/learning/transliterate/Training_bn/weights/Training_bn_model.pth')

output = bntrans.predict('aami', topk=10)
print(output)
# ['আমি', 'আমী', 'অ্যামি', 'আমিই', 'এমি', 'আমির', 'আমিদ', 'আমই', 'আমে', 'আমিতে']