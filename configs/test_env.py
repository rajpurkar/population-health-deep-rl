import time

class config():
    # env config
    state_shape         = (10, 1, 1)
    max_steps           = 5
    
    # sampling
    neccessary_queries  = 3
    num_classes         = neccessary_queries + 1
    
    #exploration
    no_repeats          = False
    no_sample_repeats   = False
    random_tie_break    = True

    # reward config
    correctAnswerReward = 10.
    wrongAnswerReward   = -1.
    queryReward         = -2.

    # training
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False
    high             = 1.

    # output config
    output_path  = "results/env_test/" + str(int(time.time())) + '/'
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 100
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 50000
    log_freq          = 100
    eval_freq         = 5000
    soft_epsilon      = 0.

    # nature paper hyper params
    nsteps_train       = 1000000
    batch_size         = 32
    buffer_size        = 100000
    target_update_freq = 1000
    gamma              = 0.999
    learning_freq      = 1
    state_history      = 1
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.001
    eps_nsteps         = nsteps_train/2
    learning_start     = 500
