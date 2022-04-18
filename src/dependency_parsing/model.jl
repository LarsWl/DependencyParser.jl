export Model
export write_to_file!, predict, loss_function, update_model!

import .ArcEager: transitions_number, set_labels!

using Flux
using JLD2

#=
  Input layer dimension: batch_size * embeddings_size
  hidden layer weights dimension(h): hidden_size * (batch_size * embeddings_size)
  bias dineansion: hidden_size
  output layer weights dimensions: labels_num * hidden_size
=#
mutable struct Model
  embeddings::Matrix{Float32}
  hidden_layer::Dense
  output_layer::Dense
  chain::Chain
  word_ids::Dict{String, Integer}
  tag_ids::Dict{String, Integer}
  label_ids::Dict{String, Integer}

  function Model(
    embeddings::Matrix{Float32},
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

    loaded_embeddings, embedding_ids = cache_data(read_embeddings_file, "tmp/cache", "path", embeddings_file)
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

    model = new(embeddings, Dense(hidden_layer, hidden_bias), Dense(output_layer, false), Chain(), word_ids, tag_ids, label_ids)
    model.chain = form_chain(model)

    model
  end
end

function cache_data(func::Function, filename::String, path::String, args...; kwargs...)
  local prepared = nothing
  if isfile(filename)
    jldopen(filename, "r") do file
      if haskey(file, path)
        prepared = file[path]
      end
    end
  end
  if isnothing(prepared)
    prepared = func(args...; kwargs...)
    jldopen(filename, "a+") do file
      file[path] = prepared
    end
  end
  return prepared
end

function form_chain(model::Model)
  hidden_layer_calculation = function (input::Vector{Integer})
    result = zeros(Float32, length(model.hidden_layer.weight[:, begin]))
    batch_size = length(input)
    embeddings_size = length(model.embeddings[begin, :])

    for i = 1:batch_size
      x = model.embeddings[input[i], :]
      offset = (i - 1) * embeddings_size
      W_slice = model.hidden_layer.weight[:, (offset + 1) : (offset + embeddings_size)]

      result += (W_slice * x + model.hidden_layer.bias) .^ 3
    end

    result
  end

  Chain(
    hidden_layer_calculation,
    model.output_layer,
    softmax
  )
end

function predict(model::Model, batch::Vector{Integer})
  model.chain(batch)
end

function update_model!(model::Model, batch::Vector{Integer}, gold::Vector{Float32})
  loss(x, y) = loss_function(model, x, y)

  params = Flux.params(model.embeddings, model.hidden_layer.weight, model.output_layer.weight)
  opt = Flux.Descent(0.1)

  Flux.train!(loss, params, [(batch, gold)], opt)
end

function loss_function(model::Model, batch::Vector{Integer}, gold::Vector{Float32})
  # Flux.Losses.crossentropy(predict(model, batch), gold)

  sum(.- sum(gold .* logsoftmax(predict(model, batch))))
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

    write(file, "$(length(model.word_ids)) $(length(model.tag_ids)) $(length(model.label_ids)) $(embeddings_size)\n")
    known_entities = vcat(collect(model.word_ids), collect(model.tag_ids), collect(model.label_ids))

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
