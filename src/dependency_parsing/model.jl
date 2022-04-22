export Model
export write_to_file!, predict, loss_function, update_model!, update_model!_2

import .ArcEager: transitions_number, set_labels!, execute_transition, zero_cost_transitions, transition_costs
import .ArcEager: GoldState, FORBIDDEN_COST

using Flux
using Zygote
using JLD2
import LinearAlgebra: norm

#=
  Input layer dimension: batch_size * embeddings_size
  hidden layer weights dimension(h): hidden_size * (batch_size * embeddings_size)
  bias dineansion: hidden_size
  output layer weights dimensions: labels_num * hidden_size
=#
mutable struct Model
  embeddings::Matrix{Float64}
  hidden_layer::Dense
  output_layer::Dense
  chain::Chain
  word_ids::Dict{String, Integer}
  tag_ids::Dict{String, Integer}
  label_ids::Dict{String, Integer}
  optimizer

  function Model(
    embeddings::Matrix{Float64},
    hidden_layer::Dense,
    output_layer::Dense,
    word_ids::Dict{String, Integer},
    tag_ids::Dict{String, Integer},
    label_ids::Dict{String, Integer},
  )
    model = new(embeddings, hidden_layer, output_layer, Chain(), word_ids, tag_ids, label_ids)
    model.chain = form_chain(model)

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
    model.output_layer = Dense(settings.hidden_size, transitions_number(system), bias=false)
    model.chain = form_chain(model)
    
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

    model = new(embeddings, Dense(hidden_layer, hidden_bias), Dense(output_layer, false), Chain(), word_ids, tag_ids, label_ids)
    model.chain = form_chain(model)

    model
  end
end

function form_chain(model::Model)
  Chain(
    input -> calculate_hidden(model, input) .^ 3,
    model.output_layer,
    softmax
  )
end

function calculate_hidden(model, input::Vector{Integer})
  result = zeros(Float64, length(model.hidden_layer.weight[:, begin]))
    batch_size = length(input)
    embeddings_size = length(model.embeddings[begin, :])

    for i = 1:batch_size
      x = model.embeddings[input[i], :]
      offset = (i - 1) * embeddings_size
      W_slice = model.hidden_layer.weight[:, (offset + 1) : (offset + embeddings_size)]

      result += (W_slice * x + model.hidden_layer.bias)
    end

    result
end

function predict(model::Model, batch::Vector{Integer})
  model.chain(batch)
end

function predict_transition(model::Model, settings::Settings, system::ParsingSystem, config::Configuration)
  form_batch(model, settings, config) |> 
    batch -> predict(model, batch) |>
    findmax |>
    max_score_wiht_index -> system.transitions[max_score_wiht_index[end]]
end

function update_model_by_corenlp!(model::Model, batch::Vector{Integer}, costs::Vector{Int64})
  grads = take_gradient(model, batch, costs)
  gold = map(costs) do cost 
    if cost == FORBIDDEN_COST
      -1
    elseif cost <= 0
      1
    else
      0
    end
  end |> softmax
  
  Flux.update!(model.optimizer, grads.params, grads)

  println("LOSS: $(loss_function(model, batch, gold, grads.params))")
end

function update_model!(model::Model, batch::Vector{Integer}, costs::Vector{Int64})
  gold = map(costs) do cost 
    if cost == FORBIDDEN_COST
      -1
    elseif cost <= 0
      1
    else
      0
    end
  end |> softmax

  params = Flux.params([
    model.embeddings,
    model.hidden_layer.weight,
    model.hidden_layer.bias,
    model.output_layer.weight
  ])
  loss(x, y) = loss_function(model, x, y, params)
  
  Flux.train!(loss, params, [(batch, gold)], model.optimizer)

  println("LOSS: $(loss_function(model, batch, gold, params))")
end

function loss_function(model::Model, batch::Vector{Integer}, gold::Vector{Float64}, params)
  chain = Chain(
    input -> calculate_hidden(model, input) .^ 3,
    model.output_layer,
  )
  scores = chain(batch)
  reg_weight = 1e-7

  sqnorm(x) = sum(abs2, x)

  Flux.Losses.logitcrossentropy(scores, gold) + sum(sqnorm, params) * (reg_weight / 2)
end

function train!(model::Model, system::ParsingSystem, settings::Settings, connlu_sentences::Vector{ConnluSentence})
  iterations = 10
  model.optimizer = ADAM()

  for i = 1:iterations
    for connlu_sentence in connlu_sentences
      sentence = zip(connlu_sentence.token_doc.tokens, connlu_sentence.pos_tags) |> collect |> Sentence

      println("Предложение: $(connlu_sentence.string_doc.text)")

      config = Configuration(sentence)

      while !is_terminal(config)
        gold_state = GoldState(connlu_sentence.gold_tree, config, system)

        predicted_transition = predict_transition(model, settings, system, config)
        zero_transitions = zero_cost_transitions(gold_state)

        if !(predicted_transition in zero_transitions)
          batch = form_batch(model, settings, config)
          costs = transition_costs(gold_state)

          update_model!(model, batch, costs)
        end

        length(zero_transitions) == 0 && continue
        transition = rand(zero_transitions)

        execute_transition(config, transition, system)
      end
    end
  end
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

    hidden_size = length(model.hidden_layer.weight[:, begin])
    labels_num = length(model.output_layer.weight[:, begin])
    batch_size = Int32(length(model.hidden_layer.weight[begin, :]) / embeddings_size)

    write(file, "$(batch_size) $(hidden_size) $(labels_num)\n")

    for i = 1:hidden_size
      write(file, join(model.hidden_layer.weight[i, :], " "))
      write(file, "\n")
    end

    write(file, join(model.hidden_layer.bias, " "))
    write(file, "\n")

    for i = 1:labels_num
      write(file, join(model.output_layer.weight[i, :], " "))
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

function get_word_id(model::Model, word::String)
  haskey(model.word_ids, word) ? model.word_ids[word] : model.word_ids[UNKNOWN_TOKEN]
end

function get_tag_id(model::Model, tag::String)
  haskey(model.tag_ids, tag) ? model.tag_ids[tag] : model.tag_ids[UNKNOWN_TOKEN]
end

function get_label_id(model::Model, label::String)
  model.label_ids[label]
end

function take_gradient(model, batch, costs)
  sum1 = 0.0
  sum2 = 0.0
  hidden = calculate_hidden(model, batch)
  cube_hidden = hidden .^ 3
  scores = model.output_layer(cube_hidden)
  max_score = findmax(scores)[begin]
  norm_coef = 1e6
  reg_weight = 1e-8
  gold = map(costs) do cost 
    if cost == FORBIDDEN_COST
      -1
    elseif cost <= 0
      1
    else
      0
    end
  end

  hidden_weight_grad = zeros(Float64, size(model.hidden_layer.weight))
  hidden_bias_grad = zeros(Float64, size(model.hidden_layer.bias))
  output_weight_grad = zeros(Float64, size(model.output_layer.weight))
  embeddings_grad = zeros(Float64, size(model.embeddings))

  cube_hidden_grad = zeros(Float64, size(cube_hidden))
  
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
      delta = -(gold[index] - scores[index] / sum2) / norm_coef;
      
      output_weight_grad[index, :] += (cube_hidden .* delta)
      output_weight_grad[index, :] += output_weight_grad[index, :] .* reg_weight
      cube_hidden_grad += (delta .* model.output_layer.weight[index, :])
    end
  end

  hidden_bias_grad = cube_hidden_grad .* 3  + (hidden .^ 2);
  hidden_bias_grad += hidden_bias_grad .* reg_weight

  foreach(enumerate(batch)) do (index, feature)
    embedding = model.embeddings[feature, :]
    embeddings_size = length(embedding)
    offset = (index - 1) * embeddings_size

    for node_index = 1:length(hidden)
      hidden_weight_grad[node_index, (offset + 1) : (offset + embeddings_size)] += embedding .* hidden_bias_grad[node_index]
      embeddings_grad[feature, :] += model.hidden_layer.weight[node_index, (offset + 1) : (offset + embeddings_size)] .* hidden_bias_grad[node_index]
    end
  end

  hidden_weight_grad += hidden_weight_grad .* reg_weight
  embeddings_grad += embeddings_grad .* reg_weight

  params = Flux.params(model.embeddings, model.hidden_layer.weight, model.output_layer.weight, model.hidden_layer.bias)

  Zygote.Grads(
    IdDict(
      model.embeddings => embeddings_grad,
      model.hidden_layer.weight => hidden_weight_grad,
      model.hidden_layer.bias => hidden_bias_grad,
      model.output_layer.weight => output_weight_grad
    ),
    params
  )
end
