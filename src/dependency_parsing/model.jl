export Model
export write_to_file!, predict, loss_function, update_model!, update_model!_2

import .ArcEager: transitions_number, set_labels!, execute_transition, zero_cost_transitions, transition_costs, gold_scores, is_transition_valid
import .ArcEager: GoldState, FORBIDDEN_COST

using Flux, CUDA
using Zygote
using JLD2
using DependencyParserTest

#=
  Input layer dimension: batch_size * embeddings_size
  hidden layer weights dimension(h): hidden_size * (batch_size * embeddings_size)
  bias dineansion: hidden_size
  output layer weights dimensions: labels_num * hidden_size
=#

# CUDA.allowscalar(false)

mutable struct Model
  embeddings
  hidden_layer::Dense
  output_layer::Dense
  word_ids::Dict{String, Integer}
  tag_ids::Dict{String, Integer}
  label_ids::Dict{String, Integer}
  gpu_available
  hidden_accumulator

  function Model(
    embeddings,
    hidden_layer::Dense,
    output_layer::Dense,
    word_ids::Dict{String, Integer},
    tag_ids::Dict{String, Integer},
    label_ids::Dict{String, Integer},
  )
    model = new(embeddings, hidden_layer, output_layer, word_ids, tag_ids, label_ids)
    model.gpu_available = CUDA.functional()
    model
  end

  function Model(settings::Settings, system::ParsingSystem, embeddings_file::String, connlu_sentences::Vector{ConnluSentence})
    model = new()

    loaded_embeddings, embedding_ids = read_embeddings_file(embeddings_file)
    embeddings_size = length(loaded_embeddings[begin, :])
    if embeddings_size != settings.embeddings_size
      ArgumentError("Incorrect embeddings dimensions. Given: $(embeddings_size). In settings: $(settings.embeddings_size)") |> throw
    end
  
    set_corpus_data!(model, connlu_sentences)
    set_labels!(system, model.label_ids |> keys |> collect)

    model.embeddings = rand(Float64, length(model.word_ids) + length(model.tag_ids) + length(model.label_ids), embeddings_size)
    match_embeddings!(model.embeddings, loaded_embeddings, embedding_ids, model.word_ids |> keys |> collect)
  
    model.hidden_layer = Dense(settings.batch_size * embeddings_size, settings.hidden_size)
    model.output_layer = Dense(settings.hidden_size, transitions_number(system), bias=false, sigmoid)
    model.gpu_available = CUDA.functional()
    
    model
  end

  function Model(model_file::String)
    lines = readlines(model_file)
    info_line = lines[begin]
    words_count, tags_count, labels_count, embeddings_size = split(info_line, " ") |> info_values -> map(value -> parse(Int32, value), info_values)
    total_ids_count =  words_count + tags_count + labels_count

    word_ids = Dict{String, Integer}()
    tag_ids = Dict{String, Integer}()
    label_ids = Dict{String, Integer}()
    embeddings = Matrix{Float64}(undef, total_ids_count, embeddings_size)

    # read all embeddings for words, tags and labels
    for i = 1:total_ids_count
      line = lines[i + 1]
      entity, embedding = split(line) |> values -> [values[begin], map(value -> parse(Float64, value), values[begin+1:end])]
      embeddings[i, :] = embedding

      # Depend on index define entity as word or as tag or as label
      if 1 <= i <= words_count
        word_ids[entity] = i
      elseif words_count < i <= words_count + tags_count
        tag_ids[entity] = i
      else
        label_ids[entity] = i
      end
    end

    info_line = lines[total_ids_count + 2]
    batch_size, hidden_size, labels_num = split(info_line) |> info_values -> map(value -> parse(Int32, value), info_values)

    hidden_layer = Matrix{Float64}(undef, hidden_size, batch_size * embeddings_size)
    output_layer = Matrix{Float64}(undef, labels_num, hidden_size)

    # read hidden layer weights and bias
    for i = 1:hidden_size
      line = lines[i + total_ids_count + 2]
      hidden_layer[i, :] = split(line) |> values -> map(value -> parse(Float64, value), values)
    end
    hidden_bias_line = lines[begin + total_ids_count + hidden_size + 2]
    hidden_bias = split(hidden_bias_line) |> values -> map(value -> parse(Float64, value), values)

    # read softmax layer weights
    for i = 1:labels_num
      line = lines[i + total_ids_count + hidden_size + 3]
      output_layer[i, :] = split(line) |> values -> map(value -> parse(Float64, value), values)
    end

    hidden_accumulator = zeros(Float32, hidden_size)

    model = new(embeddings, Dense(hidden_layer, hidden_bias), Dense(output_layer, false, sigmoid), word_ids, tag_ids, label_ids, undef, hidden_accumulator)
    model.gpu_available = CUDA.functional()
    
    enable_cuda(model)

    model
  end
end

Flux.trainable(m::Model) = (m.embeddings, m.hidden_layer.weight, m.hidden_layer.bias, m.output_layer.weight,)

function enable_cuda(model::Model)
  if model.gpu_available
    @info "enable cuda"
    model.embeddings = cu(model.embeddings)
    model.hidden_layer = fmap(cu, model.hidden_layer)
    model.output_layer = fmap(cu, model.output_layer)
    model.hidden_accumulator = cu(model.hidden_accumulator)
  end
end

function calculate_hidden(model, input; dropout_active=false)
  fill!(model.hidden_accumulator, zero(Float32))

  embeddings_size = length(model.embeddings[begin, :])
  batch_size = length(input)
  hidden_weight = model.hidden_layer.weight

  for i in 1:batch_size
    offset = (i - 1) * embeddings_size
    hidden_slice = view(hidden_weight, :, (offset + 1) : (offset + embeddings_size))

    model.hidden_accumulator .+= hidden_slice * input[i]
  end

  model.hidden_accumulator .+= model.hidden_layer.bias

  Flux.dropout(model.hidden_accumulator, 0.5, active = dropout_active, dims = 1)
end

function predict(model::Model, batch)
  (calculate_hidden(model, batch) .^ 3) |>
    model.output_layer |>
    result -> reshape(result, 1, :)
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
    batch_emb ->(calculate_hidden(model, batch_emb, dropout_active = true) .^ 3) |>
    Flux.normalise |>
    model.output_layer |>
    result -> reshape(result, 1, :)
end


function predict_transition(model::Model, settings::Settings, system::ParsingSystem, config::Configuration)
  scores = form_batch(model, settings, config) |> batch -> take_batch_embeddings(model, batch) |>
    batch -> predict(model, batch) |> collect |>
    result -> filter(score -> score[end] > 0 && is_transition_valid(config, system.transitions[score[begin]], system), collect(enumerate(result)))

  length(scores) == 0 && return

  score_index = findmax(map(score -> score[end], scores))[end]

  system.transitions[scores[score_index][begin]]
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

function loss_function(entropy_sum, ps, settings::Settings)
  entropy_sum + L2_norm(ps, settings)
end

function L2_norm(ps, settings::Settings)
  sqnorm(x) = sum(abs2, x)

  sum(sqnorm, ps) * (settings.reg_weight / 2)
end

function transition_loss(scores, gold)
  Flux.Losses.binary_focal_loss(scores, gold, γ=2)
end

function test_transition_loss(scores, gold)
  Flux.Losses.binarycrossentropy(scores, gold)
end


function train!(model::Model, training_context::TrainingContext)
  training_context.optimizer = ADAM()

  training_context.test_dataset = Flux.DataLoader(build_dataset(model, training_context.test_connlu_sentences[begin:begin+800], training_context), batchsize=training_context.settings.sample_size, shuffle=true)
  
  raw_dataset = build_dataset(model, training_context.connlu_sentences, training_context)
  train_datasets = Flux.DataLoader(raw_dataset, batchsize=Int64(round(length(raw_dataset) / 10)), shuffle=true)

  train_epoch = () -> begin
    for dataset in train_datasets
      samples = Flux.DataLoader(dataset, batchsize=training_context.settings.sample_size, shuffle=true)

      update_model!(model, samples, training_context)

      GC.gc()
      CUDA.reclaim()

      @info "Dataset ends, start test"
      test_training_scores(model, training_context)

      GC.gc()
      CUDA.reclaim()
    end
  end

  @info "Start training"
  Flux.@epochs 500 train_epoch()
end

# THERE IS A TEMP CODE USED FOR TRAINING THAT I RUN IN REPL

# train_samples = Flux.DataLoader(build_dataset(model, training_context.connlu_sentences, training_context), batchsize=training_context.settings.sample_size, shuffle=true)
# train_samples.data |> length
# println(sum(sent -> test_loss(model, sent, training_context), sentences_batch) / (sentences_batch |> length))
# for i in 1:100
#   train_samples = Flux.DataLoader(build_dataset(model, sentences_batch, training_context), batchsize=training_context.settings.sample_size)
#   grads = take_grads(model, train_samples, training_context)
#   update_model!(model, grads, training_context)
# end
# println(sum(sent -> test_loss(model, sent, training_context), sentences_batch) / (sentences_batch |> length))

# using Random
# dataset = first(train_datasets)
# data = first(Flux.DataLoader(dataset, batchsize=training_context.settings.sample_size, shuffle=true))

# context = training_context
# sample = data
# sample = weighted_sample(model, data, training_context)

# @show sample[1][2]

# findall(d -> d[end][begin, begin] == 1, data) |> length
# println(map(d -> findmax(d[2][begin, :])[end], data))
# loss = (sample) -> begin
#     sum(sample) do (batch, gold, weight)
#       predict_train(model, batch) |> scores -> weight * transition_loss(scores, gold)
#     end + L2_norm(ps, training_context.settings)
#   end
# ps = Flux.params(model)

# batch, gold = data[begin]

# gold
# cu(rand(200, 100)) * batch[1]
# predict_train(model, batch)
# predict(model, take_batch_embeddings(model, batch))

# test_transition_loss(predict(model, take_batch_embeddings(model, batch)), gold)

# loss([data[begin]])

# @show data[begin][2]


# [[1], [0], [1]] |> vecs -> hcat(vecs...)

# m = Dense(200, 100, bias = false, sigmoid)
# vec = rand(200)

# m(vec)
# connlu_sentence = training_context.connlu_sentences[begin + 1]
# sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence
# config = Configuration(sentence)

# t = predict_transition(model, training_context.settings, training_context.system, config)
# execute_transition(config, trans, training_context.system)
# gold_state = GoldState(connlu_sentence.gold_tree, config, training_context.system)
# gold  = transition_costs(gold_state) |> gold_scores
# config

# trans = Transition(ArcEager.Shift(), "")

# println(gold[begin, :])

# batch = form_batch(model, training_context.settings, config) |> b -> take_batch_embeddings(model, b)
# findmax(predict(model, batch)[begin, :])

# train_samples.data |> length
# training_context.system.transitions[7]
# findmax(gold[begin, :])
# println(loss(data))
# for i in 1:1
#   Flux.train!(loss, ps, [data], training_context.optimizer)
# end
# println(loss(data))

# loss_dict = Dict()

# data = first(training_context.test_dataset)
# foreach(data) do (batch, gold, weight)
#   correct = findall(e -> e != 0, gold[begin, :])
#   if haskey(loss_dict, correct)
#     loss_dict[correct] += predict_train(model, batch) |> scores -> weight * transition_loss(scores, gold)
#   else
#     loss_dict[correct] = predict_train(model, batch) |> scores -> weight * transition_loss(scores, gold)
#   end
# end
# @info sort(collect(loss_dict), by = pair -> pair[end])

# loss_dict_2 = Dict()

# false_positives = zeros(size(data[begin][2]))
# false_negatives = zeros(size(data[begin][2]))
# false_positives_with_transition = zeros(size(data[begin][2]))
# false_negatives_with_transition = zeros(size(data[begin][2]))

# foreach(training_context.test_dataset.data) do (batch, gold)
#   scores = take_batch_embeddings(model, batch) |> batch_emb -> predict(model, batch_emb) |> collect
#   cpu_gold = gold |> collect

#   for i in 1:length(scores[begin, :])
#     for j in 1:length(scores[:, begin])
#       if scores[j, i] > 0.33 && round(cpu_gold[j, i], RoundDown) == 0
#         false_positives[j, i] += 1
#       end
#       if scores[j, i] < 0.33 && round(cpu_gold[j, i], RoundDown) == 1
#         false_negatives[j, i] += 1
#       end
#     end
#   end
# end
# @info sort(collect(loss_dict), by = pair -> pair[end])
# @info weight_matrix

# @show false_negatives
# @show false_positives
# system.transitions[18]
# loss_dict_3 = Dict()

# foreach(data) do (batch, gold, weight)
#   scores = predict_train(model, batch)

  
#   for i in 1:length(scores[begin, :])
#     if haskey(loss_dict, i)
#       loss_dict[i] += Flux.crossentropy(scores[:, i], gold[:, i])
#     else
#       loss_dict[i] = Flux.crossentropy(scores[:, i], gold[:, i])
#     end
#   end
# end
# @info sort(collect(loss_dict_2), by = pair -> pair[end])

function build_dataset(model, sentences_batch, training_context)
  train_samples = []
  frequency = Dict()
  mutex = ReentrantLock()
  sent_mutex = ReentrantLock()
  freq_mutex = ReentrantLock()
  sentences_number = 0

  parse_gold_tree = (connlu_sentence) -> begin
    batch = []

    sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

    config = Configuration(sentence)
    process_config(model, config, connlu_sentence.gold_tree, training_context, 0, mutex, batch)

    for (_, gold) in batch
      gold_correct = findall(e -> e == 1, gold)
  
      lock(freq_mutex) do 
        if haskey(frequency, gold_correct)
          frequency[gold_correct] += 1
        else
          frequency[gold_correct] = 1
        end
      end
    end

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

  @info "Do some upsample...."

  train_samples = map(train_samples) do sample
    gold = sample[end]
    gold_correct = findall(e -> e != 0, gold)

    repeat_number = if frequency[gold_correct] < 300
      7
    elseif frequency[gold_correct] < 700
      5
    elseif frequency[gold_correct] < 1000
      3
    else
      1
    end

    frequency[gold_correct] += repeat_number - 1
    
    if model.gpu_available
      gold = cu(gold)
    end

    [(sample[begin], gold) for i in 1:repeat_number]
  end |> Iterators.flatten |> collect

  @info frequency
  @info "Dataset builded..."

  train_samples
end

function process_config(model, config, gold_tree, context, transition_number, mutex, train_samples)
  is_terminal(config) && return
  transition_number >= LIMIT_TRANSITIONS_NUMBER && return

  gold_state = GoldState(gold_tree, config, context.system)
  gold  = transition_costs(gold_state) |> gold_scores

  zero_transitions = zero_cost_transitions(gold_state)
  length(zero_transitions) == 0 && return

  batch = form_batch(model, context.settings, config)

  lock(mutex) do 
    push!(train_samples, (batch, gold))
  end
  
  if rand(1:100) in 1:(context.beam_coef * 100)
    foreach(zero_transitions) do transition
      next_config = deepcopy(config)
      execute_transition(next_config, transition, context.system)

      process_config(model, next_config, gold_tree, context, transition_number + 1, mutex, train_samples)
    end
  else
    transition = rand(zero_transitions)
    execute_transition(config, transition, context.system)

    process_config(model, config, gold_tree, context, transition_number + 1, mutex, train_samples)
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
    take_batch_embeddings(model, batch) |>
    batch_emb -> predict(model, batch_emb) |>
      scores -> test_transition_loss(scores, gold)
  end + L2_norm(ps, context.settings)
end

#=
File structure
known_words_count known_pos_tags_count known_labels_count embeddings_size
words embeddings
pos_tags embeddings
labels embeddings
batch_size hidden_size label_nums
hidden weights
hidden bias
softmax weights
=#
function write_to_file!(model::Model, filename::String)
  embeddings = model.embeddings |> collect
  hidden_weight = model.hidden_layer.weight |> collect
  hidden_bias = model.hidden_layer.bias |> collect
  output_weight = model.output_layer.weight |> collect

  open(filename, "w") do file
    embeddings_size = length(embeddings[begin, :])

    sort_by_value(pair) = pair[end]
    write(file, "$(length(model.word_ids)) $(length(model.tag_ids)) $(length(model.label_ids)) $(embeddings_size)\n")
    known_entities = vcat(
      sort(collect(model.word_ids), by = sort_by_value),
      sort(collect(model.tag_ids), by = sort_by_value),
      sort(collect(model.label_ids), by = sort_by_value)
    )

    foreach(known_entities) do (entity, entity_id)
      embedding = embeddings[entity_id, :]
      write(file, "$(entity) $(join(embedding, " "))\n")
    end

    hidden_size = length(hidden_weight[:, begin])
    labels_num = length(output_weight[:, begin])
    batch_size = Int32(length(hidden_weight[begin, :]) / embeddings_size)

    write(file, "$(batch_size) $(hidden_size) $(labels_num)\n")

    for i = 1:hidden_size
      write(file, join(hidden_weight[i, :], " "))
      write(file, "\n")
    end

    write(file, join(hidden_bias, " "))
    write(file, "\n")

    for i = 1:labels_num
      write(file, join(output_weight[i, :], " "))
      write(file, "\n")
    end
  end
end

function read_embeddings_file(filename::String)
  lines = readlines(filename)
  words_count, dimension = split(lines[begin]) |> (numbers -> map(number -> parse(Int64, number), numbers))
  deleteat!(lines, 1)

  embeddings = zeros(Float64, words_count, dimension)
  embedding_ids = Dict{String, Integer}()

  foreach(enumerate(lines)) do (index, line)
    splitted = split(line, " ")
    word = line[begin]
    embedding_ids[string(word)] = index

    for i = 1:dimension
      embeddings[index, i] = parse(Float64, splitted[i + 1])
    end
  end

  [embeddings, embedding_ids]
end

function match_embeddings!(embeddings::Matrix{Float64}, loaded_embeddings::Matrix{Float64}, embedding_ids::Dict{String, Integer}, known_words::Vector{String})
  embeddings_size = size(embeddings)[end]
  
  foreach(enumerate(known_words)) do (index, word)
    embedding_id = if haskey(embedding_ids, word)
      embedding_ids[word]
    elseif haskey(embedding_ids, lowercase(word))
      embedding_ids[lowercase(word)]
    else
      0
    end

    if embedding_id > 0
      embeddings[index, :] = loaded_embeddings[embedding_id, :]
    else
      embeddings[index, :] = [rand(-100:100) / 10000.0 for i in 1:embeddings_size]
    end
  end
end

function set_corpus_data!(model::Model, connlu_sentences::Vector{ConnluSentence})
  seq_id = Sequence()

  model.word_ids = Dict{String, Integer}()
  model.tag_ids = Dict{String, Integer}()
  model.label_ids = Dict{String, Integer}()

  corpus = map(conllu_sentence -> conllu_sentence.token_doc, connlu_sentences) |> Corpus
  update_lexicon!(corpus)


  lexicon(corpus) |> keys |> collect |> words -> foreach(word -> model.word_ids[word] = next!(seq_id), words)
  model.word_ids[UNKNOWN_TOKEN]= next!(seq_id)
  model.word_ids[NULL_TOKEN]= next!(seq_id)
  model.word_ids[ROOT_TOKEN]= next!(seq_id)

  map(conllu_sentence -> conllu_sentence.pos_tags, connlu_sentences) |>
    Iterators.flatten |>
    collect |>
    unique |>
    tags -> foreach(tag -> model.tag_ids[tag] = next!(seq_id), tags)
  model.tag_ids[UNKNOWN_TOKEN]= next!(seq_id)
  model.tag_ids[NULL_TOKEN]= next!(seq_id)
  model.tag_ids[ROOT_TOKEN]= next!(seq_id)

  map(conllu_sentence -> conllu_sentence.gold_tree.nodes, connlu_sentences) |>
    Iterators.flatten |> 
    collect |>
    nodes -> map(node -> node.label, nodes) |>
    unique |>
    labels -> foreach(label -> model.label_ids[label] = next!(seq_id), labels)
  model.label_ids[NULL_TOKEN]= next!(seq_id)
end

#=
while batch_size only is 48 there is structure of batch
1-18 - word_ids
19-36 - tag_ids
37-48 - label_ids
=#
const POS_OFFSET = 18
const LABEL_OFFSET = 36
const STACK_OFFSET = 6
const STACK_NUMBER = 6

function form_batch(model::Model, settings::Settings, config::Configuration)
  batch = zeros(Integer, settings.batch_size)

  word_id_by_word_index(word_index::Integer) = get_token(config, word_index) |> token -> get_word_id(model, token)
  tag_id_by_word_index(word_index::Integer) = get_tag(config, word_index) |> tag -> get_tag_id(model, tag)
  label_id_by_word_index(word_index::Integer) = get_label(config, word_index) |> label -> get_label_id(model, label)

  # add top three stack elements and top three buffers elems with their's tags
  for i = 1:3
    stack_word_index = get_stack_element(config, i)
    buffer_word_index = get_buffer_element(config, i)

    batch[i] = word_id_by_word_index(stack_word_index)
    batch[i + POS_OFFSET] = tag_id_by_word_index(stack_word_index)
    batch[i + 3] = word_id_by_word_index(buffer_word_index)
    batch[i + POS_OFFSET + 3] = tag_id_by_word_index(buffer_word_index)
  end

  #=
    Add: 
    The first and second leftmost / rightmost children of the top two words on the stack and
    The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack
  =#
  for stack_id = 1:2
    stack_word_index = get_stack_element(config, stack_id)

    set_word_data_by_index_with_offset = function (word_index::Integer, additional_offset)
      batch[STACK_OFFSET + (stack_id - 1) * STACK_NUMBER + additional_offset] = word_id_by_word_index(word_index)
      batch[STACK_OFFSET + POS_OFFSET + (stack_id - 1) * STACK_NUMBER + additional_offset] = tag_id_by_word_index(word_index)
      batch[LABEL_OFFSET + (stack_id - 1) * STACK_NUMBER + additional_offset] = label_id_by_word_index(word_index)
    end

    get_left_child(config.tree, stack_word_index) |> word_index -> set_word_data_by_index_with_offset(word_index, 1)
    get_right_child(config.tree, stack_word_index) |> word_index -> set_word_data_by_index_with_offset(word_index, 2)
    get_left_child(config.tree, stack_word_index, child_number=2) |> word_index -> set_word_data_by_index_with_offset(word_index, 3)
    get_right_child(config.tree, stack_word_index, child_number=2) |> word_index -> set_word_data_by_index_with_offset(word_index, 4)
    get_left_child(config.tree, stack_word_index) |> 
      word_index -> get_left_child(config.tree, word_index) |>
      word_index -> set_word_data_by_index_with_offset(word_index, 5)
    get_right_child(config.tree, stack_word_index) |> 
      word_index -> get_right_child(config.tree, word_index) |>
      word_index -> set_word_data_by_index_with_offset(word_index, 6)
  end
  
  batch
end

function take_batch_embeddings(model, batch)
  take_embedding(word_index::Integer) = view(model.embeddings, word_index, :)
  batch = map(take_embedding, batch)

  batch
end

function get_word_id(model::Model, word::String)
  haskey(model.word_ids, word) ? model.word_ids[word] : model.word_ids[UNKNOWN_TOKEN]
end

function get_tag_id(model::Model, tag::String)
  haskey(model.tag_ids, tag) ? model.tag_ids[tag] : model.tag_ids[UNKNOWN_TOKEN]
end

function get_label_id(model::Model, label::String)
  label == EMPTY_LABEL ? model.label_ids[NULL_TOKEN] : model.label_ids[label]
end

function try_unsafe_free(x)
  hasmethod(CUDA.unsafe_free!, Tuple{typeof(x)}) && CUDA.unsafe_free!(x)
end