from random import shuffle
counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, ',': 0}
with open('amazon_total.csv') as fh:
    for line in fh:
        counts[line[-2]] += 1
del(counts[','])
max_entries = min(counts.values())
for c in counts:
    counts[c] = 0
all_lines = []
with open('amazon_total.csv', 'r') as in_file:
    for line in in_file:
        try:
            if counts[line[-2]] < max_entries:
                all_lines.append(line)
                counts[line[-2]] += 1
        except:
            pass
shuffle(all_lines)
with open('balanced_data.csv', 'w') as fh:
    for line in all_lines:
        fh.write(line)