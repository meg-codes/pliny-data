import csv
import glob

for i in range(1, 10):
    counts = {}
    for input in glob.glob('letters_lemmatized/%s_*.txt' % i):
        with open(input) as fp:
            text = fp.read()
            words = text.split()
            for word in words:
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
    with open('book_%s.csv' % i, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['word', 'frequency'])
        tuples = [(k, v) for k, v in counts.items()]
        tuples.sort(key=lambda x: x[1], reverse=True)
        for tup in tuples:
            writer.writerow(tup)
