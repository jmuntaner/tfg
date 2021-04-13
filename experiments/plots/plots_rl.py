import sys, re
NUM_EPOCHS = 40
def get_losses(log):
    epochs = []
    loss = 0
    valid_loss = 0
    for l in log.splitlines()[-NUM_EPOCHS*2-1:]:
        line = re.split(' |=',l)
        if len(line) < 1:
            continue
        if line[0] == 'Validation':
            for index, token in enumerate(line):
                if token == 'BLEU':
                    valid_loss = line[index+1]
                    epochs.append({'loss': loss, 'valid_loss': valid_loss})
                    break
        elif line[0] == 'Epoch':
            for index, token in enumerate(line):
                if token == 'BLEU':
                    loss = line[index+1]
                    break
    return epochs

import matplotlib.pyplot as plt
PATH = '/home/usuaris/veu/joan.muntaner/experiments/outputs/deen_rl2.log'
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
    print(max(train[1:]))
    print(valid)
    print(max(valid[1:]))
    print(len(train), len(valid))
    plt.plot(list(range(0, len(train))),train,label='train')
    plt.plot(list(range(0, len(train))),valid,label='valid')
    plt.xticks(list(range(0, NUM_EPOCHS+1, 5)))
    plt.yticks(list(range(35, 50, 2)))
    plt.xlim(0)
    plt.legend()
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel='BLEU Score')
    plt.title('RL training in LSTM in IWSLT14 DE_EN 40 epochs')
    plt.savefig('RL_LSTM_IWSLT_40_epochs.png')
    #plt.show()

if __name__ == "__main__":
    main()