import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    print(dataset)
    train_synset = []
    with open(f'{dataset}/train.txt') as in_f:
        for line in in_f:
            l = line.strip().split()
            train_synset.append(l[0])
            train_synset.append(l[2])

    train_synset = set(train_synset)

    valid_synset = []
    test_synset = []

    valid_instance_count = 0
    test_instance_count = 0

    valid_unknown_instance_count = 0
    test_unknown_instance_count = 0

    with open(f'{dataset}/valid.txt') as in_f:
        for line in in_f:
            l = line.strip().split()
            valid_synset.append(l[0])
            valid_synset.append(l[2])
            valid_instance_count += 1
            if l[0] not in train_synset or l[2] not in train_synset:
                valid_unknown_instance_count += 1

    with open(f'{dataset}/test.txt') as in_f:
        for line in in_f:
            l = line.strip().split()
            test_synset.append(l[0])
            test_synset.append(l[2])
            test_instance_count += 1
            if l[0] not in train_synset or l[2] not in train_synset:
                test_unknown_instance_count += 1

    valid_synset = set(valid_synset)
    test_synset = set(test_synset)

    valid_unknown_count = 0
    test_unknown_count = 0

    for synset in valid_synset:
        if synset not in train_synset:
            valid_unknown_count += 1

    for synset in test_synset:
        if synset not in train_synset:
            test_unknown_count += 1

    print(f'Summary for dataset {dataset}')
    print(f'Number of unknown synsets in valid (total)   : {valid_unknown_count} ({len(valid_synset)})')
    print(f'Proptn of unknown synsets in valid           : {valid_unknown_count / len(valid_synset)}')
    print(f'Number of unknown instances in valid (total) : {valid_unknown_instance_count} ({valid_instance_count})')
    print(f'Proptn of unknown instances in valid         : {valid_unknown_instance_count / valid_instance_count}')

    print(f'Number of unknown synsets in test  (total)  : {test_unknown_count } ({len(test_synset)})')
    print(f'Proptn of unknown synsets in test           : {valid_unknown_count / len(test_synset)}')
    print(f'Number of unknown instances in test (total) : {test_unknown_instance_count} ({test_instance_count})')
    print(f'Proptn of unknown instances in test         : {test_unknown_instance_count / test_instance_count}')
