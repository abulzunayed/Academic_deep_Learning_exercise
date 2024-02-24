# from generator import ImageGenerator
#
# label_path = './Labels.json'
# file_path = './exercise_data/'
# gen = ImageGenerator(file_path, label_path, 18, [32, 32, 3], rotation=True, mirroring=False, shuffle=False)
# gen.show()

import pattern

a = pattern.Circle((100,100), 5, (25,25))
a.draw()