

a_file = open("colors.txt", "r")
list_of_lines = a_file.readlines()
list_of_lines[-2] = "magenta\n"

a_file = open("colors.txt", "w")
a_file.writelines(list_of_lines)
a_file.close()