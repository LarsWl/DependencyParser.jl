export Model
export write_to_file!, predict

import .ArcEager: transitions_number, set_labels!

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
    model.output_layer = Dense(settings.hidden_size, transitions_number(system), bias=false)
    model.hidden_accumulator = zeros(Float32, settings.hidden_size)

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

    model = new(embeddings, Dense(hidden_layer, hidden_bias), Dense(output_layer, false), word_ids, tag_ids, label_ids, undef, hidden_accumulator)
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

function calculate_hidden(model, input)
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

  model.hidden_accumulator
end

function predict(model::Model, batch)
  take_batch_embeddings(model, batch) |>
    batch_emb -> (calculate_hidden(model, batch_emb) .^ 3) |>
    model.output_layer |>
    softmax
end

function predict_transition(model::Model, settings::Settings, system::ParsingSystem, config::Configuration)
  scores = form_batch(model, settings, config) |>
    batch -> predict(model, batch) |> collect |>
    result -> filter(score -> score[end] > 0 && is_transition_valid(config, system.transitions[score[begin]], system), collect(enumerate(result)))

  length(scores) == 0 && return

  score_index = findmax(map(score -> score[end], scores))[end]

  system.transitions[scores[score_index][begin]]
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
1-12 - word_ids
13-24 - tag_ids
25-32 - label_ids
=#
const POS_OFFSET = 12
const LABEL_OFFSET = 24
const STACK_OFFSET = 4
const STACK_NUMBER = 4

function form_batch(model::Model, settings::Settings, config::Configuration)
  batch = zeros(Integer, settings.batch_size)

  word_id_by_word_index(word_index::Integer) = get_token(config, word_index) |> token -> get_word_id(model, token)
  tag_id_by_word_index(word_index::Integer) = get_tag(config, word_index) |> tag -> get_tag_id(model, tag)
  label_id_by_word_index(word_index::Integer) = get_label(config, word_index) |> label -> get_label_id(model, label)

  # add top two stack elements and top two buffers elems with their's tags
  for i = 1:2
    stack_word_index = get_stack_element(config, i)
    buffer_word_index = get_buffer_element(config, i)

    batch[i] = word_id_by_word_index(stack_word_index)
    batch[i + POS_OFFSET] = tag_id_by_word_index(stack_word_index)
    batch[i + 2] = word_id_by_word_index(buffer_word_index)
    batch[i + POS_OFFSET + 2] = tag_id_by_word_index(buffer_word_index)
  end

  #=
    Add: 
    The first and second leftmost / rightmost children of the first stack and buffer element
  =#

  set_word_data_by_index_with_offset = function (word_index::Integer, additional_offset)
    batch[STACK_OFFSET + additional_offset] = word_id_by_word_index(word_index)
    batch[STACK_OFFSET + POS_OFFSET + additional_offset] = tag_id_by_word_index(word_index)
    batch[LABEL_OFFSET + additional_offset] = label_id_by_word_index(word_index)
  end

  set_children_data = (config_word_index, offset_number) -> begin
    get_left_child(config.tree, config_word_index) |> word_index -> set_word_data_by_index_with_offset(word_index, offset_number * STACK_NUMBER + 1)
    get_right_child(config.tree, config_word_index) |> word_index -> set_word_data_by_index_with_offset(word_index, offset_number * STACK_NUMBER + 2)
    get_left_child(config.tree, config_word_index, child_number=2) |> word_index -> set_word_data_by_index_with_offset(word_index, offset_number * STACK_NUMBER + 3)
    get_right_child(config.tree, config_word_index, child_number=2) |> word_index -> set_word_data_by_index_with_offset(word_index, offset_number * STACK_NUMBER + 4)
  end

  set_children_data(get_stack_element(config, 1), 0)
  set_children_data(get_buffer_element(config, 1), 1)
  
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
