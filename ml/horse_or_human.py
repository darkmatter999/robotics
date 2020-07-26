# This Computer Vision dataset classifies if a given image shows a human or a horse.
# It consists of 1000 images, 500 for each category.
# The 'human' category consists of examples of the following groups (each male and female, adult and kid):
# Caucasian - Black - East Asian - South Asian
# For each of these four subcategories there is an equal amount of adult and kid images, separated between male and female.
# 500 images in total / 4 = 125 / 4 ~ 31 images per subcategory
# Naming: hu-cman, hu-cfan, hu-cmkn, hu-cfkn ... for black just start with 'hu-b', for East Asian with 'hu-e', for South Asian with 'hu-s'
# above, 'n' stands for the current image number, i.e. pics from 1 to n
# likewise, 'hu' stands for 'human'

# The horse category consists of examples of the following groups:
# Big - Small (Pony)
# 500 images in total / 2 = 250 images per subcategory
# Naming: ho-bn, ho-sn
