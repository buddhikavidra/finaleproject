import os
DATA_DIR='D:/reserch/law casess/training and testingdata/Intellectual Property' #Industrial Disputes Act/205.txt
lines=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
xxx = []
def delete_multiple_lines(original_file, line_numbers):
    """In a file, delete the lines at line number in given list"""
    is_skipped = False
    counter = 0
    # Create name of dummy / temporary file
    dummy_file = original_file + '.bak'
    # Open original file in read only mode and dummy file in write mode
    with open(original_file, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Line by line copy data from original file to dummy file
        for line in read_obj:
            # If current line number exist in list then skip copying that line
            if counter not in line_numbers:
                write_obj.write(line)
            else:
                is_skipped = True
            counter += 1
    # If any line is skipped then rename dummy file as original file
    if is_skipped:
        os.remove(original_file)
        print('suc')
        os.rename(dummy_file, original_file)
    else:
        os.remove(dummy_file)
        print('no')

import os
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        print(os.path.join(DATA_DIR, file))
        xx = os.path.join(DATA_DIR, file)
        xxx.append(xx)
        delete_multiple_lines(xx,lines)