'''
Pre-process Logs File
extract a pure sentence (remove date, speaker, log_id)
'''

import os


def sub_write(reader, writer):
    '''
    :param reader: file which wanna read it
    :param writer: file which wanna write it
    :return:
    '''
    line = reader.readline()
    while True:
        # break loop when we found a empty string
        if line == '':
            break

        line = line.strip().split(sep='\t')
        if len(line) < 4:
            continue

        writer.write(line[-1])
        writer.write('\n')

        # read new input line
        line = reader.readline()


main_read_path = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Dataset/dialog/3'
main_write_path = '/Users/jamemamjame/Computer-Sci/_chula course/SENIOR PROJECT/Dataset/processed_dialog/'

# collect file name from main_read_path directory
list_filename = os.listdir(main_read_path)
w_filename = os.path.join(main_write_path, 'logs3.txt')

# index for cutting list_filename
strt, end = 0, 20000

# using 'a' mode for append write (not new write)
with open(w_filename, 'a') as writer:
    for i, filename in enumerate(list_filename[strt: end]):
        r_filename = os.path.join(main_read_path, filename)
        try:
            with open(r_filename, 'r') as reader:
                sub_write(reader=reader, writer=writer)
        except:
            print('error at %s(%d)' % (filename, strt + i))
