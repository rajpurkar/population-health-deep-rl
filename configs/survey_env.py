import time

class config():
    # env config
    max_steps           = 4
    num_classes         = 2

    correctAnswerReward = 5.
    wrongAnswerReward   = -5.
    queryReward         = -0.1
    queryRewardMap      = { 'Region': -0.01 }

    #exploration
    no_repeats          = False
    no_sample_repeats   = False
    random_tie_break    = True
    force_pred          = False

    # training
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False
    high             = 1.

    # output config
    output_path  = "results/survey_env_test/" + "lower_overall_rewards" + '/'
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 2000
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 50000
    log_freq          = 100
    eval_freq         = 1000
    soft_epsilon      = 0.

    # nature paper hyper params
    nsteps_train       = 100000#0
    batch_size         = 32
    buffer_size        = 100000
    target_update_freq = 1000
    gamma              = 1.0
    learning_freq      = 1
    state_history      = 1
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = int(nsteps_train/1.1)
    eps_begin          = 1
    eps_end            = 0.001
    eps_nsteps         = int(nsteps_train/1.1)
    learning_start     = 500
