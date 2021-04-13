import sys, re
NUM_EPOCHS = 30
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
                if token == 'loss':
                    valid_loss = line[index+1]
                    epochs.append({'loss': loss, 'valid_loss': valid_loss})
                    break
        elif line[0] == 'Epoch':
            for index, token in enumerate(line):
                if token == 'loss':
                    loss = line[index+1]
                    break
    return epochs

import matplotlib.pyplot as plt
PATH = '/home/usuaris/veu/joan.muntaner/experiments/outputs/neen_mle_alt.log'
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
    plt.xticks(list(range(5, NUM_EPOCHS+1, 5)))
    plt.yticks(list(range(0, 8, 1)))
    plt.xlim(0)
    plt.legend()
    plt.xlabel(xlabel='Epoch')
    plt.ylabel(ylabel='Cross-entropy loss')
    plt.title('Bi-LSTM with MLE in FLoRes NE_EN alternative_param set dataset')
    plt.savefig('train_flores_lstm_mle_alt.png')
    #plt.show()

if __name__ == "__main__":
    main()