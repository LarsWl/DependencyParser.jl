export train!

import .ArcEager: set_labels!, execute_transition, have_zero_cost_transitions, transition_costs, optimal_transition_index, is_transition_valid
import .ArcEager: GoldState, FORBIDDEN_COST

using Flux

# some of operators in calculate_hidden not compilable for Zygote
function calculate_hidden_train(model, input)
    result = zeros(Float32, length(model.hidden_layer.weight[:, begin]))
    if model.gpu_available
        result = cu(result)
    end

    embeddings_size = length(model.embeddings[begin, :])
    batch_size = length(input)
    hidden_weight = model.hidden_layer.weight

    for i in 1:batch_size
        offset = (i - 1) * embeddings_size
        hidden_slice = view(hidden_weight, :, (offset+1):(offset+embeddings_size))

        result += hidden_slice * input[i]
    end

    result += model.hidden_layer.bias

    Flux.dropout(result, 0.5, dims=1)
end

function train_predict_tree(model::Model, sentence::Sentence, context::TrainingContext)
    config = Configuration(sentence)
    transitions_number = 0

    while !is_terminal(config)
        transition = predict_transition(model, context.settings, context.system, config)
        transition === nothing && break
        execute_transition(config, transition, context.system)

        transitions_number += 1
        transitions_number > LIMIT_TRANSITIONS_NUMBER && break
    end

    config.tree
end

function predict_train(model::Model, batch)
    take_batch_embeddings(model, batch) |>
    batch_emb -> (calculate_hidden_train(model, batch_emb) .^ 3) |>
                 Flux.normalise |>
                 model.output_layer |>
                 Flux.normalise |>
                 softmax
end

function update_model!(model::Model, dataset, training_context::TrainingContext)
    ps = Flux.params(model)

    loss = (sample) -> begin
        sum(sample) do (batch, gold)
            predict_train(model, batch) |> scores -> transition_loss(scores, gold)
        end + L2_norm(ps, training_context.settings)
    end

    test_sample = first(dataset)

    evalcb = () -> begin
        @show test_loss(model, test_sample, training_context)
        CUDA.reclaim()
    end

    throttle_cb = Flux.throttle(evalcb, 10)

    Flux.train!(loss, ps, dataset, training_context.optimizer, cb=throttle_cb)
end

function L2_norm(ps, settings::Settings)
    sqnorm(x) = sum(abs2, x)

    sum(sqnorm, ps) * (settings.reg_weight / 2)
end

function transition_loss(scores, gold)
    Flux.Losses.focal_loss(scores, gold, γ=2)
end

function test_transition_loss(scores, gold)
    Flux.Losses.crossentropy(scores, gold)
end

function train!(model::Model, training_context::TrainingContext)
    training_context.optimizer = ADAM()

    training_context.test_dataset = Flux.DataLoader(build_dataset(model, training_context.test_connlu_sentences[begin:begin+1200], training_context), batchsize=training_context.settings.sample_size, shuffle=true)

    raw_dataset = build_dataset(model, training_context.connlu_sentences, training_context)
    train_samples = Flux.DataLoader(raw_dataset, batchsize=training_context.settings.sample_size, shuffle=true)

    train_epoch = () -> begin
        update_model!(model, train_samples, training_context)

        GC.gc()
        CUDA.reclaim()

        @info "Dataset ends, start test"
        test_training_scores(model, training_context)

        GC.gc()
        CUDA.reclaim()

    end

    @info "Start training"
    Flux.@epochs 500 train_epoch()
end

function build_dataset(model, sentences_batch, training_context)
    train_samples = []
    mutex = ReentrantLock()
    sent_mutex = ReentrantLock()
    sentences_number = 0

    parse_gold_tree = (connlu_sentence) -> begin
        batch = []

        sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

        config = Configuration(sentence)
        process_config(model, config, connlu_sentence.gold_tree, training_context, 0, mutex, batch)

        lock(sent_mutex) do
            train_samples = vcat(train_samples, batch)
            sentences_number += 1
            if sentences_number % 250 == 0
                @info "Sentences processed: $sentences_number"
            end
        end
    end


    try
        Threads.@sync begin
            @info "Build dataset..."
            for connlu_sentence in sentences_batch
                Threads.@spawn parse_gold_tree(connlu_sentence)
            end
        end
    catch err
        println(err.task.exception)
        return err
    end

    @info "Dataset builded..."

    train_samples
end

function process_config(model, config, gold_tree, context, transition_number, mutex, train_samples)
    while !is_terminal(config)
        transition_number >= LIMIT_TRANSITIONS_NUMBER && return

        gold_state = GoldState(gold_tree, config, context.system)
        have_zero_cost_transitions(gold_state) || return

        costs = transition_costs(gold_state)
        opt_tranisition_index = optimal_transition_index(costs, context.system)
        gold = Flux.onehot(opt_tranisition_index, 1:length(costs))

        if model.gpu_available
            gold = gold |> collect |> cu
        end

        batch = form_batch(model, context.settings, config)

        lock(mutex) do
            push!(train_samples, (batch, gold))
        end

        execute_transition(config, context.system.transitions[opt_tranisition_index], context.system)
    end
end

function test_training_scores(model::Model, context::TrainingContext)
    losses = []
    parsed_trees_file = "tmp/parsed_trees.txt"
    mutex = ReentrantLock()
    losses_mutex = ReentrantLock()
    parse_tree = (connlu_sentence, file) -> begin
        sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

        tree = train_predict_tree(model, sentence, context)
        tree_text = convert_to_string(tree)

        lock(mutex) do
            write(file, tree_text)
            write(file, "\n")
            write(file, "\n")
        end
    end

    parse_sample = (sample) -> begin
        loss = test_loss(model, sample, context)

        lock(losses_mutex) do
            push!(losses, loss)
        end
    end

    open(parsed_trees_file, "w") do file
        for i in 1:length(context.test_connlu_sentences)
            connlu_sentence = context.test_connlu_sentences[i]

            parse_tree(connlu_sentence, file)
        end
    end

    test_dataset = context.test_dataset |> collect

    for i in 1:length(test_dataset)
        sample = test_dataset[i]

        parse_sample(sample)
    end

    conllu_source = DependencyParserTest.Sources.ConlluSource(context.test_connlu_file)
    parser_source = DependencyParserTest.Sources.CoreNLPSource(parsed_trees_file)

    uas, las = DependencyParserTest.Benchmark.test_accuracy(conllu_source, parser_source)
    best_loss = min(losses...)
    worst_loss = max(losses...)
    avg_loss = sum(losses) / length(losses)

    open(context.training_results_file, "a") do file
        result_line = "UAS=$(uas), LAS=$(las), best_loss=$(best_loss), worst_loss=$(worst_loss), avg_loss=$(avg_loss)\n"
        write(file, result_line)
    end

    write_to_file!(model, context.model_file * "_last.txt")

    if uas > context.best_uas || (uas == context.best_uas && las > context.best_las)
        context.best_uas = uas
        context.best_las = las
        context.best_loss = avg_loss

        write_to_file!(model, context.model_file * "_best.txt")
    end
end

function test_loss(model::Model, sample, context)
    ps = Flux.params(model)

    sum(sample) do (batch, gold)
        predict(model, batch) |> scores -> test_transition_loss(scores, gold)
    end + L2_norm(ps, context.settings)
end

# TEST CODE

# data = first(training_context.test_dataset)

# batch, gold = data[begin]

# predict_train(model, batch)
# predict(model, batch)

# test_training_scores(model, training_context)

# update_model!(model, [data], training_context)

# transition_loss(predict_train(model, batch), gold)