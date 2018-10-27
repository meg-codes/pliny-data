import os
import re

BASE_FILENAME = os.path.join('letters', '%s_%s.txt')

with open('LAT1318.unicode.txt') as pliny_file:

    start = outfile = book = letter = lines = None

    # make a directory to hold the output split files
    if not os.path.isdir('letters'):
        os.mkdir('letters')
    # iterate over each line
    for line in pliny_file:
        # flag start of the text
        if line.startswith('[1.1]'):
            start = True
        # if we've hit the start of the text
        if start:
            # if we hit a new book
            new_book = re.match('\[(\d+\.\d)\]', line)
            if new_book:
                # see if it's the weird 1.1 exception and set up for writing
                # otherwise just book will report as true
                book = int(new_book.group(1).split('.')[0])
                letter = int(new_book.group(1).split('.')[1])
                if outfile:
                    outfile.close()
                if book and letter:
                    if not outfile:
                        outfile = open(BASE_FILENAME % (book, letter), 'w')
                # continue because we don't want to do anything with the
                # letter salutations
                continue

            # if we hit a new letter
            new_letter = re.match('\[(\d+)\]', line)
            if new_letter:
                # set the letter
                letter = int(new_letter.group(1))
                # repoint the file
                outfile.close()
                outfile = open(BASE_FILENAME % (book, letter), 'w')
                # continue to drop the letter salutations
                continue

            # if there's a book a letter, we're in the text of a letter
            # and want to write the line
            # otherwise let the loop continue through
            if book and letter:
                outfile.write(line)
