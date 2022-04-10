export Model

import .ArcEager: transitions_number, set_labels!

using Flux
#=
  Input layer dimension: batch_size * embeddings_size
  hidden layer weights dimension(h): hidden_size * (batch_size * embeddings_size)
  bias dineansion: hidden_size
  output layer weights dimensions: labels_num * hidden_size
=#
struct Model
  embeddings::Matrix{Float32}
  hidden_layer::Dense
  softmax_layer::Dense

  function Model(settings::Settings, system::ParsingSystem, embeddings_file::String, connlu_sentences::Vector{ConnluSentence})
    loaded_embeddings, embedding_ids = read_embeddings_file(embeddings_file)
    embeddings_size = length(loaded_embeddings[begin, :])
    if embeddings_size != settings.embedding_size
      ArgumentError("Incorrect embeddings dimensions. Given: $(embeddings_size). In settings: $(settings.embedding_size)") |> throw
    end
  
    corpus = map(conllu_sentence -> conllu_sentence.token_doc, connlu_sentences) |> Corpus
    update_lexicon!(corpus)
  
    known_words = lexicon(corpus) |> keys |> collect
    known_tags = map(conllu_sentence -> conllu_sentence.pos_tags, connlu_sentences) |> Iterators.flatten |> collect |> unique
    known_labels = map(conllu_sentence -> conllu_sentence.gold_tree.nodes, connlu_sentences) |>
      Iterators.flatten |> 
      collect |>
      nodes -> map(node -> node.label, nodes) |>
      unique
    set_labels!(system, known_labels)
  
    embeddings = rand(Float32, length(known_words) + length(known_tags) + length(known_labels), embeddings_size)
    match_embeddings!(embeddings, loaded_embeddings, embedding_ids, known_words)
  
    hidden_layer = Dense(settings.batch_size * embeddings_size, settings.hidden_size)
    softmax_layer = Dense(settings.hidden_size, transitions_number(system))

    new(embeddings, hidden_layer, softmax_layer)
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

function write_to_file(filename::String)
end

