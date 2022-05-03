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

CUDA.allowscalar(false)

mutable struct Model
  embeddings
  hidden_layer::Dense
  output_layer::Dense
  word_ids::Dict{String, Integer}
  tag_ids::Dict{String, Integer}
  label_ids::Dict{String, Integer}
  gpu_available

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

    model.embeddings = rand(Float32, length(model.word_ids) + length(model.tag_ids) + length(model.label_ids), embeddings_size)
    match_embeddings!(model.embeddings, loaded_embeddings, embedding_ids, model.word_ids |> keys |> collect)
  
    model.hidden_layer = Dense(settings.batch_size * embeddings_size, settings.hidden_size)
    model.output_layer = Dense(settings.hidden_size, transitions_number(system), bias=false)
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
    embeddings = Matrix{Float32}(undef, total_ids_count, embeddings_size)

    # read all embeddings for words, tags and labels
    for i = 1:total_ids_count
      line = lines[i + 1]
      entity, embedding = split(line) |> values -> [values[begin], map(value -> parse(Float32, value), values[begin+1:end])]
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

    hidden_layer = Matrix{Float32}(undef, hidden_size, batch_size * embeddings_size)
    output_layer = Matrix{Float32}(undef, labels_num, hidden_size)

    # read hidden layer weights and bias
    for i = 1:hidden_size
      line = lines[i + total_ids_count + 2]
      hidden_layer[i, :] = split(line) |> values -> map(value -> parse(Float32, value), values)
    end
    hidden_bias_line = lines[begin + total_ids_count + hidden_size + 2]
    hidden_bias = split(hidden_bias_line) |> values -> map(value -> parse(Float32, value), values)

    # read softmax layer weights
    for i = 1:labels_num
      line = lines[i + total_ids_count + hidden_size + 3]
      output_layer[i, :] = split(line) |> values -> map(value -> parse(Float32, value), values)
    end

    model = new(embeddings, Dense(hidden_layer, hidden_bias), Dense(output_layer, false), word_ids, tag_ids, label_ids)
    model.gpu_available = CUDA.functional()

    model
  end
end

function calculate_hidden(model, input; dropout_active=false)
  result = zeros(Float32, length(model.hidden_layer.weight[:, begin]))

  embeddings_size = length(model.embeddings[begin, :])
  batch_size = length(input)

  # hidden_weight = Flux.dropout(model.hidden_layer.weight, 0.5, active = dropout_active, dims = 1)
  mask = if dropout_active
    [rand(0:1) for i in size(model.hidden_layer.weight)[begin]]
  else
    ones(size(model.hidden_layer.weight)[begin])
  end

  if model.gpu_available
    result = cu(result)
    mask = cu(mask)
  end

  for i = 1:batch_size
    offset = (i - 1) * embeddings_size
    result .+= view(model.hidden_layer.weight, :, (offset + 1) : (offset + embeddings_size)) .* mask * input[i]
  end

  result .+= model.hidden_layer.bias

  CUDA.unsafe_free!(mask)
  #CUDA.unsafe_free!(hidden_weight)

  result
end

function params!(model::Model)
  Flux.params([
    model.embeddings,
    model.hidden_layer.weight,
    model.hidden_layer.bias,
    model.output_layer.weight
  ])
end

function predict(model::Model, batch)
  (calculate_hidden(model, batch) .^ 3) |>
    model.output_layer |>
    softmax
end

function train_predict_tree(model::Model, sentence::Sentence, context::TrainingContext)
  config = Configuration(sentence)

  while !is_terminal(config)
    transition = predict_transition(model, context.settings, context.system, config)
    is_transition_valid(config, transition, context.system) || break
    execute_transition(config, transition, context.system)
  end

  config.tree
end

function predict_train(model::Model, batch)
  result = (calculate_hidden(model, batch, dropout_active = true) .^ 3) |>
    model.output_layer |>
    softmax

  result
end


function predict_transition(model::Model, settings::Settings, system::ParsingSystem, config::Configuration)
  form_batch(model, settings, config) |> batch -> take_batch_embeddings(model, batch) |>
    batch ->predict(model, batch) |>
    scores -> findmax(scores) |>
    max_score_with_index -> system.transitions[max_score_with_index[end]]
end

function update_model!(model::Model, grads, training_context::TrainingContext)
  println("Begin update model....")

  params = params!(model)

  Flux.update!(training_context.optimizer, params, grads)

  for param in params
    CUDA.unsafe_free!(grads[param])
  end
  CUDA.reclaim()
end

function loss_function(entropy_sum, params, settings::Settings)
  entropy_sum + L2_norm(params, settings)
end

function L2_norm(params, settings::Settings)
  sqnorm(x) = sum(abs2, x)

  sum(sqnorm, params) * (settings.reg_weight / 2)
end

function transition_loss(scores, gold)
  Flux.Losses.binary_focal_loss(scores, gold, γ=3)
end

function train!(model::Model, training_context::TrainingContext)
  training_context.optimizer = ADAGrad(0.01)
  train_samples = []
  model.gpu_available = CUDA.functional()

  if model.gpu_available
    println("enable cuda")
    model.embeddings = cu(model.embeddings)
    model.hidden_layer = fmap(cu, model.hidden_layer)
    model.output_layer = fmap(cu, model.output_layer)
  end

  train_epoch = () -> begin
    sentences_batches = Flux.DataLoader(training_context.connlu_sentences, batchsize=500)

    for sentences_batch in sentences_batches
      train_samples = Flux.DataLoader(build_dataset(model, sentences_batch, training_context), batchsize=training_context.settings.sample_size)
      grads = take_grads(model, train_samples, training_context)
      update_model!(model, grads, training_context)
    end

    println("Epocha ends, start test")
    test_training_scores(model, training_context)
  end

  Flux.@epochs 500 train_epoch()
end

function take_grads(model, dataset, context)
  params = params!(model)
  loss(x, y) = transition_loss(x, y) + L2_norm(params, context.settings)

  init_grad_value = (size) -> begin
    if model.gpu_available
      CUDA.zeros(Float32, size)
    else
      zeros(Float32, size)
    end
  end
  grads = IdDict(
    model.embeddings => size(model.embeddings) |> init_grad_value,
    model.hidden_layer.weight => size(model.hidden_layer.weight) |> init_grad_value,
    model.hidden_layer.bias => size(model.hidden_layer.bias) |> init_grad_value,
    model.output_layer.weight => size(model.output_layer.weight) |> init_grad_value
  )

  mutex = ReentrantLock()
  merge_grads = (computed_grads) -> begin
    for param in params
      grad = computed_grads[param]
      lock(mutex) do 
        grads[param] .+= (grad + grad .* (context.settings.reg_weight))
      end

      hasmethod(CUDA.unsafe_free!, Tuple{typeof(grad)}) && CUDA.unsafe_free!(grad)
    end
  end

  compute_sample = (sample) -> begin
    println("Thread $(Threads.threadid()) begin compute...")

    temp_grads = IdDict(
      model.embeddings => size(model.embeddings) |> init_grad_value,
      model.hidden_layer.weight => size(model.hidden_layer.weight) |> init_grad_value,
      model.hidden_layer.bias => size(model.hidden_layer.bias) |> init_grad_value,
      model.output_layer.weight => size(model.output_layer.weight) |> init_grad_value
    )
    data_processed = 0

    for data in sample
      scores = data[begin]
      gold = data[end]
      gs = gradient(params) do
        loss(scores, gold)
      end
      CUDA.unsafe_free!(scores)
      CUDA.unsafe_free!(gold)
      for param in params
        grad = if hasmethod(CUDA.unsafe_free!, Tuple{typeof(gs[param])})
          gs[param]
        else
          cu(gs[param])
        end

        temp_grads[param] .+= grad

        CUDA.unsafe_free!(grad)
      end
      gs = nothing

      data_processed += 1

      if data_processed % 500 == 0
        CUDA.reclaim()
        GC.gc(false)
      end

      if data_processed % 5000 == 0
        CUDA.memory_status()
        println("\nThread $(Threads.threadid()) computed additional 5000 data")
      end
    end

    merge_grads(temp_grads)

     CUDA.reclaim()
  end

  try
    Threads.@sync begin
      for sample in dataset
        Threads.@spawn compute_sample(sample)
      end
    end
  catch err
    println(err.task.exception)
    return err
  end

  Zygote.Grads(grads, params)
end

function build_dataset(model, sentences_batch, training_context)
  train_samples = []
  mutex = ReentrantLock()
  sent_mutex = ReentrantLock()
  sentences_number = 0
  parse_gold_tree = (connlu_sentence) -> begin
    sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence
    transitions_number = 0
    config = Configuration(sentence)

    while !is_terminal(config)
      gold_state = GoldState(connlu_sentence.gold_tree, config, training_context.system)

      zero_transitions = zero_cost_transitions(gold_state)
      length(zero_transitions) == 0 && break

      batch = form_batch(model, training_context.settings, config) |> batch -> take_batch_embeddings(model, batch)
      scores = predict_train(model, batch)

      gold  = transition_costs(gold_state) |> gold_scores
      if model.gpu_available
        gold = cu(gold)
      end

      lock(mutex) do 
        push!(train_samples, (scores, gold))
      end

      transition = rand(zero_transitions)
      execute_transition(config, transition, training_context.system)
      transitions_number += 1
      transitions_number >= LIMIT_TRANSITIONS_NUMBER && break
    end

    lock(sent_mutex) do 
      sentences_number += 1
      if sentences_number % 100 == 0
        CUDA.reclaim()
        CUDA.memory_status()
        println("\nSentences processed: $sentences_number")
      end
    end
  end


  try
    Threads.@sync begin
      println("Build dataset...")
      for connlu_sentence in sentences_batch
        parse_gold_tree(connlu_sentence)
      end
    end
  catch err
    println(err.task.exception)
    return err
  end

  println("Dataset builded...")
  GC.gc(false)
  CUDA.reclaim()
  CUDA.memory_status()

  train_samples
end

function test_training_scores(model::Model, context::TrainingContext)
  losses = []
  parsed_trees_file = "tmp/parsed_trees.txt"
  mutex = ReentrantLock()
  parse_tree = (connlu_sentence, file) -> begin
    sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

    tree = train_predict_tree(model, sentence, context)
    tree_text = convert_to_string(tree)
    avg_loss = test_loss(model, connlu_sentence, context)

    lock(mutex) do 
      write(file, tree_text)
      write(file, "\n")
      write(file, "\n")

      push!(losses, avg_loss)
    end
  end

  Threads.@sync begin
    open(parsed_trees_file, "w") do file
      foreach(context.test_connlu_sentences) do connlu_sentence
        parse_tree(connlu_sentence, file)
      end
    end
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

  if uas > context.best_uas || (uas == context.best_uas && las > context.best_las)
    context.best_uas = uas
    context.best_las = las
    context.best_loss = avg_loss

    write_to_file!(model, context.model_file)
  end
end

function test_loss(model::Model, connlu_sentence::ConnluSentence, training_context::TrainingContext)
  params = params!(model)
  chain = Chain(
    input -> calculate_hidden(model, input) .^ 3,
    model.output_layer,
    softmax
  )
  sentence_sample = []

  sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence
  config = Configuration(sentence)
  transitions_number = 0

  while !is_terminal(config)
    gold_state = GoldState(connlu_sentence.gold_tree, config, training_context.system)

    predicted_transition = predict_transition(model, training_context.settings, training_context.system, config)
    zero_transitions = zero_cost_transitions(gold_state)
    length(zero_transitions) == 0 && break

    if !(predicted_transition in zero_transitions)
      batch = form_batch(model, training_context.settings, config) |> batch -> take_batch_embeddings(model, batch)
      gold  = transition_costs(gold_state) |> gold_scores

      if model.gpu_available
        gold = cu(gold)
      end

      push!(sentence_sample, (batch, gold))
    end
    
    transition = rand(zero_transitions)

    execute_transition(config, transition, training_context.system)

    transitions_number += 1

    transitions_number >= LIMIT_TRANSITIONS_NUMBER && break
  end

  entropy_sum = sum(sentence_sample) do (batch, gold)
    transition_loss(chain(batch), gold)
  end

  loss_function(entropy_sum, params, training_context.settings)
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
  hidden_weight = model.hidden_layer.weight |> collect
  hidden_bias = model.hidden_layer.bias |> collect
  output_weight = model.output_layer.weight |> collect

  open(filename, "w") do file
    embeddings_size = length(model.embeddings[begin, :])

    sort_by_value(pair) = pair[end]
    write(file, "$(length(model.word_ids)) $(length(model.tag_ids)) $(length(model.label_ids)) $(embeddings_size)\n")
    known_entities = vcat(
      sort(collect(model.word_ids), by = sort_by_value),
      sort(collect(model.tag_ids), by = sort_by_value),
      sort(collect(model.label_ids), by = sort_by_value)
    )

    foreach(known_entities) do (entity, entity_id)
      embedding = model.embeddings[entity_id, :]
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

  embeddings = zeros(Float32, words_count, dimension)
  embedding_ids = Dict{String, Integer}()

  foreach(enumerate(lines)) do (index, line)
    splitted = split(line, " ")
    word = line[begin]
    embedding_ids[string(word)] = index

    for i = 1:dimension
      embeddings[index, i] = parse(Float32, splitted[i + 1])
    end
  end

  [embeddings, embedding_ids]
end

function match_embeddings!(embeddings::Matrix{Float32}, loaded_embeddings::Matrix{Float32}, embedding_ids::Dict{String, Integer}, known_words::Vector{String})
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

  if model.gpu_available
    # map(cu, batch)
  else
    batch
  end

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


function take_corenlp_gradient(model, batch, costs, grads_dict, context)
  sum1 = 0.0
  sum2 = 0.0
  hidden = calculate_hidden(model, take_batch_embeddings(model, batch), dropout_active=true)
  cube_hidden = hidden .^ 3
  scores = model.output_layer(cube_hidden) |> collect
  costs = costs |> collect
  max_score = findmax(scores)[begin]
  gold = map(costs) do cost 
    if cost == FORBIDDEN_COST
      -1
    elseif cost <= 0
      1
    else
      0
    end
  end

  hidden_weight_grad = grads_dict[model.hidden_layer.weight]
  hidden_bias_grad = grads_dict[model.hidden_layer.bias]
  output_weight_grad = grads_dict[model.output_layer.weight]
  embeddings_grad = grads_dict[model.embeddings]

  cube_hidden_grad = CUDA.zeros(Float32, size(cube_hidden))
  
  foreach(enumerate(costs)) do (index, cost)
    if cost != FORBIDDEN_COST
      scores[index] = exp(scores[index] - max_score)
      if cost <= 0
        sum1 += scores[index]
      end
      sum2 += scores[index]
    end
  end

  for index = 1:length(costs)
    if costs[index] != FORBIDDEN_COST
      delta = -(gold[index] - scores[index] / sum2) / context.settings.sample_size;

      output_weight = view(model.output_layer.weight, index, :)
      
      view(output_weight_grad, index, :) .+= (cube_hidden .* delta)
      cube_hidden_grad .+= (delta .* output_weight)
    end
  end

  hidden_grad = cube_hidden_grad .* 3  + (hidden .^ 2);
  hidden_bias_grad .+= hidden_grad

  embeddings_size = context.settings.embeddings_size

  for index = 1:context.settings.batch_size
    word_index = batch[index]
    embedding = cu(model.embeddings[word_index, :])
    offset = (index - 1) * embeddings_size

    cpu_hidden_grad = hidden_grad |> collect

    for node_index = 1:length(hidden)
      view(hidden_weight_grad, node_index, (offset + 1) : (offset + embeddings_size)) .+= embedding .* cpu_hidden_grad[node_index]
      view(embeddings_grad, word_index, :) .+= view(model.hidden_layer.weight, node_index, (offset + 1) : (offset + embeddings_size)) .* cpu_hidden_grad[node_index]
    end
  end

  CUDA.unsafe_free!(hidden)
  CUDA.unsafe_free!(cube_hidden)
  CUDA.unsafe_free!(cube_hidden_grad)
  CUDA.unsafe_free!(hidden_grad)
end