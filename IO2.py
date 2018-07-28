file = open('text','r')


lines = file.read()
# no_of_lines = 6
# lines = ""
# print('input please:')
# for i in range(no_of_lines):
#     lines+=input('')+"\n"

print(lines, "here")
# for i, char in enumerate(lines):
#     print(i,char)
#     if char == ' ':
#         if (lines[i-1] == '-') | (lines[i+1] == '-'):
#             lines[i] = ''
#print(''.join(lines))
print()

for i, char in enumerate(lines):
    if char == ' ':
        if (lines[i-1] == '-') | (lines[i+1] == '-'):
            lines[i] = ''
    if char.isnumeric():
        if lines[i-1].isnumeric():
            pass
        elif lines[i+1].isnumeric():
            new = (int(lines[i])*10 + int(lines[i+1]) + 11)
            if new > 70:
                new = new%70
                if new <= 0:
                    new = 70+new
            lines[i] = str(new)
            lines[i+1] = ''
        else:
            new = int(char) + 11
            lines[i] = str(new)
#lines = ' '.join(lines)
print(''.join(lines))

'''
for i, char in enumerate(lines):
    if char.isnumeric():
        if lines[i-1].isnumeric():
            pass
        elif lines[i+1].isnumeric():
            new = (int(lines[i])*10 + int(lines[i]) + 12) % 70
            lines[i] = str(new)
            lines[i+1] = ''
        else:
            new = int(char) + 12
            lines[i] = str(new)
#lines = ' '.join(lines)
print(''.join(lines))
'''
print('1'.isnumeric())
print(int('1') + 12)