# Exploring Reinforcement Learning in NMT
Implementation reference {#sec:appendix}
========================

In this appendix we provide an overview of our scripts and some
examples.

Overview
--------

We recommend reading the Fairseq documentation[^1] as well as taking a
look at both Fairseq[^2] and simple-nmt[^3] repositories in order to
have a better understanding of our code. It must be noted that our
scripts are intended for running in a GPU Cluster with Slurm.\
\
The relevant files in our code are:

-   `./fairseq/fairseq/criterions/v2.py`: file that implements the
    REINFORCE algorithm, it is derived from Tianchun’s github[^4] with a
    few modifications.

-   `./fairseq/examples/translation/prepare-iwslt14.sh`: script that
    prepares the IWSLT14 DE-EN data (download, tokenization, BPE
    learning, set splitting).

-   `./fairseq/flores/download-data.sh`: script that downloads the data
    from the FLoRes dataset.

-   `./fairseq/flores/prepare-neen.sh`: script that prepares the FLoRes
    NE-EN data (tokenization, BPE learning, set splitting).

-   `./fairseq/train_baseline_smoothed.sh`: script for training a
    baseline using the Transformer with label-smoothing with the IWSLT14
    DE-EN dataset.

-   `./fairseq/generate_baseline_smoothed.sh`: script that performs
    inference and BLEU scores the model obtained by the previous script.

-   `./fairseq/train_baseline_neen.sh`: script for training a baseline
    using the Transformer with label-smoothing with the FLoRes dataset.

-   `./fairseq/generate_neen.sh`: script that performs inference and
    BLEU scores the model obtained by the previous script.

-   `./experiments/utils_preprocessing/nlp_preprocessing/detokenizer.py`:
    script that detokenizes data, useful for when we score the FLoRes
    dataset.

-   `./experiments/plots`: scripts for parsing training log files of
    simple-nmt, and generating graphs of the loss function. Adapted from
    this file[^5].

-   `./fairseq/plots`: scripts for parsing training log files of
    fairseq, and generating graphs of the loss function. Adapted from
    this file[^6].

-   `./experiments/simplenmt/simple-nmt/evaluate_generic.sh`: generic
    script that performs inference on IWSLT14 models.

-   `./experiments/simplenmt/simple-nmt/evaluate_generic_flores.sh`:
    generic script that performs inference on FLoRes models.

-   `./experiments/simplenmt/simple-nmt/score_generic.sh`: script that
    BLEU scores the translations obtained by the `evaluate_generic.sh`
    script, that is, from the IWSLT14 dataset.

-   `./experiments/simplenmt/simple-nmt/score_generic_flores.sh`: script
    that BLEU scores the translations obtained by the
    `evaluate_generic_flores.sh` script, that is, from the FLoRes
    dataset.

-   IWSLT14 training scripts, located at
    `./experiments/simplenmt/simple-nmt/`:

    -   `deen_train_mle.sh`: trains a model with Bi-LSTM using MLE loss.

    -   `deen_train_mle_alt.sh`: trains a model with Bi-LSTM using MLE
        loss and an alternative set of parameters.

    -   `deen_train_mrt.sh`: trains a model with Bi-LSTM using MRT loss.

-   FLoRes training scripts, located at
    `./experiments/simplenmt/simple-nmt/`:

    -   `neen_train_mle.sh`: trains a model with Bi-LSTM using MLE loss.

    -   `neen_train_mle_alt.sh`: trains a model with Bi-LSTM using MLE
        loss and an alternative set of parameters.

    -   `neen_train_mrt.sh`: trains a model with Bi-LSTM using MRT loss.

    -   `neen_train_mrt_alt.sh`: trains a model with Bi-LSTM using MRT
        loss starting from the alternative MLE model.

Usage
-----

First of all we need to fulfill the requirements to be able to execute
the code, (see `requirements.txt`. Then we need to download and prepare
the data. For the IWSLT14 case we proceed by:

``` {.bash language="bash" caption="IWSLT14" data="" preparation=""}
# Download and prepare the data
cd fairseq/examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

For the FLoRes data we execute (from within the flores folder in fairseq
(`./fairseq/flores`):

``` {.bash language="bash" caption="FLoRes" data="" preparation=""}
bash download-data.sh
bash prepare-neen.sh
```

### Training a Transformer model

In order to training a Transformer model we proceed by executing the
following

``` {.bash language="bash"}
sbatch train_baseline_smoothed.sh
```

We have to take into account that as we are executing this models in a
slurm cluster, the output will appear in the file we set as output in
the line `#SBATCH –output=`. We also can choose the location of the
checkpoints of the training by changing the `CP_DIR` variable. Finally,
it’s also important to set the Fairseq directory correctly into the
variable `FAIRSEQ_DIR`. Once we have finished training, we can perform
inference and scoring of our best model (located in the checkpoints
folder we specified) by executing(again, we must be mindful of the
checkpoints, fairseq and output directories and files:

``` {.bash language="bash"}
sbatch generate_baseline_smoothed.sh
```

By now, we can check the output file and see what BLEU score our model
has achieved.

### Training a simple-nmt model

The pipeline for training the models here is
`train -> evaluate -> score`. We will show how to train an MLE model and
train MRT upon it. In order to execute the train scripts properly, as
before one has to assure to input the correct directories for the data,
the model destination folder and the output log file. For example, let’s
train a FLoRes model:

``` {.bash language="bash"}
sbatch neen_train_mle_alt.sh
```

Now, looking at the output log file we can select the best model (in our
case the one with the lowest validation loss) and perform inference. We
can use the `evaluate_generic_flores.sh` script for that purpose, we
only need to specify the model path and the output file for the
translated sentences.

``` {.bash language="bash"}
sbatch evaluate_generic_flores.sh
```

By now, we already have our hyps, we just need to detokenize the
sentences and compare them to the references in order to get a BLEU
Score. We use the `score_generic_flores.sh` script for that purpose. We
have to modify the HYPS\_DIR variable so that it points to the
translated sentences file we obtained from the evaluate script. Note
that we have to adequately indicate the Fairseq and preprocessing
directories as well as making sure our environments fulfill all
requirements[^7].

``` {.bash language="bash"}
sbatch score_generic_flores.sh
```

At this point, we could check the obtained BLEU Score in the output log
file. Now if we wanted to train a MRT model upon our trained MLE model,
we simply would have to follow again with the
`train -> evaluate -> score` pipeline, although this time we should use
the mrt version of the train script (`neen_train_mrt_alt.sh`).

[^1]: <https://fairseq.readthedocs.io/en/latest/>

[^2]: <https://github.com/pytorch/fairseq>

[^3]: <https://github.com/kh-kim/simple-nmt>

[^4]: <https://github.com/TianchunH97/fairseq-rl>

[^5]: <https://github.com/jordiae/fairseq-factored/blob/master/plot/plots.py>

[^6]: <https://github.com/jordiae/fairseq-factored/blob/master/plot/plots.py>

[^7]: The script uses two different conda environments because in our
    settings the simple-nmt library and the fairseq library were in
    different folders under different conda environments.
