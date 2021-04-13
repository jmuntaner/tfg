import sys
def get_losses(log):
    epochs = []
    loss = 0
    valid_loss = 0
    for l in log.splitlines()[1:]:
        if len(l.split()) < 6:
            continue
        if l.split()[5] == 'valid':
            for index, token in enumerate(l.split()):
                if token == 'loss':
                    valid_loss = l.split()[index+1]
                    break
        elif l.split()[5] == 'train':
            for index, token in enumerate(l.split()):
                if token == 'loss':
                    loss = l.split()[index+1]
                    epochs.append({'loss': loss, 'valid_loss': valid_loss})
                    break
    return epochs

import matplotlib.pyplot as plt
PATH = '/home/usuaris/veu/joan.muntaner/tfg/new/logs/flores/train_baseline_neen.log'
def main():
    #logs = get_logs(sys.argv[1:])
    #l = get_losses(log)
    l = get_losses(open(PATH, 'r').read())
    train = [None]
    valid = [None]
    for x in l:
        train.append(float(x['loss']))
        valid.append(float(x['valid_loss']))
    print(l)
    print(len(l))
    x, y = zip(*l)  # unpack a list of pairs into two tuples
    print(x)
    print(train)
    print(min(train[1:]))
    print(valid)
    print(min(valid[1:]))
    print(len(train), len(valid))
    plt.plot(list(range(0, len(train))),train,label='train')
    plt.plot(list(range(0, len(train))),valid,label='valid')
    plt.xticks(list(range(5, 100, 10)))
    plt.yticks(list(range(0, 11, 1)))
    plt.xlim(0)
    plt.legend()
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel='Cross-entropy loss')
    plt.title('Baseline Transformer in IWSLT14 DE_EN dataset')
    #plt.savefig('plots/trainfloresbase.png')
    #plt.show()

if __name__ == "__main__":
    main()