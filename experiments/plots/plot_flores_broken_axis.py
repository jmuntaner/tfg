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
PATH = '/home/usuaris/veu/joan.muntaner/experiments/outputs/neen_rl_alt.log'
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
    f, (ax, ax2) = plt.subplots(2,1, sharex=True)
    ax.plot(list(range(0, len(train))),train,label='train')
    ax.plot(list(range(0, len(train))),valid,label='valid')
    ax2.plot(list(range(0, len(train))),train,label='train')
    ax2.plot(list(range(0, len(train))),valid,label='valid')
    ax.set_ylim(55, 75)
    ax2.set_ylim(7,9)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # plt.plot(list(range(0, len(train))),train,label='train')
    # plt.plot(list(range(0, len(train))),valid,label='valid')
    # plt.xticks(list(range(5, NUM_EPOCHS+1, 5)))
    # plt.yticks(list(range(1, 60, 8)))
    # plt.xlim(0)
    ax.legend()
    plt.xlabel(xlabel='Epoch')
    # plt.ylabel(ylabel='BLEU Score')
    #f.supxlabel('Epoch',fontsize='medium')
    f.supylabel('BLEU Score', fontsize='medium', x=0.05)
    plt.suptitle('RL training in LSTM in FLoRes NE_EN v2')
    plt.savefig('RL_neen_LSTM_FLoRes_RL_2_40epochs.png')
    #plt.show()

if __name__ == "__main__":
    main()